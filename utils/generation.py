import math
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import os, sys, contextlib, traceback, re, ast
from gurobipy import Model, GRB, quicksum
from scipy.special import betaln, gammaln

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from func_timeout import func_timeout, FunctionTimedOut

open_tag = "<<"
close_tag = ">>"



@contextlib.contextmanager
def suppress_output(to_devnull: bool = True):
    stream = open(os.devnull, 'w')
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield
    if to_devnull:
        stream.close()


def parse(llm_output, key, evaluate = False):
    """
    Parse the output of an LLM to extract content between {key} tags.
    """

    content = StrOutputParser().parse(llm_output)
    content = content.replace("```python\n", "").replace("```", "")
    matches = re.findall(rf"{open_tag}BEGIN_{key}{close_tag}(.*?){open_tag}END_{key}{close_tag}", content, re.DOTALL)
    matches = [m.strip() for m in matches]
    if evaluate:
        return ast.literal_eval(matches)
    return matches



def str_to_var(func_str, func_name):
    namespace = {}
    code_obj = compile(func_str, filename=f"{func_name}.py", mode="exec")
    with suppress_output():
        exec(code_obj, namespace)
    return namespace[func_name]




def execute_code(func_str, func_name, inputs):
    try:
        function = str_to_var(func_str, func_name)
        with suppress_output():
            output = function(*inputs)
        return (0, output)
    except Exception as e:
        orig_tb = sys.exc_info()[2]
        tb = orig_tb

        stderr = ""
        while tb:
            frame = tb.tb_frame
            code = frame.f_code
            if code.co_filename.endswith(f"{func_name}.py"):
                lineno = tb.tb_lineno
                src_lines = func_str.splitlines()

                start = max(0, lineno-3)
                end   = min(len(src_lines), lineno+2)
                stderr += f"\nError occurred in solver.py at line {lineno}:\n"
                for idx in range(start, end):
                    prefix = "→" if idx == lineno-1 else " "
                    stderr += f"{prefix} {idx+1:4d}: {src_lines[idx]}\n"

                stderr += "\nLocal variables at crash point:\n"
                for var, val in frame.f_locals.items():
                    stderr += f"  {var!r} = {val!r}\n"
                break
            tb = tb.tb_next

        stderr += "\nCaptured traceback:\n"
        stderr += traceback.format_exc()

        return (1, stderr)
    




##### Solvers #####

def generate_solvers(solvers_file, problem_data, model, n_solvers, temperature = 0.7):
    if os.path.exists(solvers_file):
        with open(solvers_file, "rb") as file:
            solvers = pickle.load(file)
    else:
        solvers = []

    n_new = n_solvers - len(solvers)
    n_valid = 0
    
    while n_valid < n_solvers:
        print(f"# solvers: {len(solvers)} --> {n_solvers}")

        structured_output = not (model in ['o3', 'o3-mini'])
        if structured_output:
            llm = ChatOpenAI(model=model, temperature=temperature)
            schema = {
                "title": "CodeGenerator",
                "description": "Structured response with complete code",
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The complete python code for the 'solve' function",
                    },
                },
                "required": ["code"]
            }
            llm = llm.with_structured_output(schema)
        else:
            llm = ChatOpenAI(model=model)

        query = "\n".join([
            "You are a code-generation agent expert in Python and gurobipy. numpy, gurobipy, pandas, shapely, and networkx are the only library you can use.",
            f"Here is the problem description:\n {problem_data['problem_description']}\n",
            f"Here is the input template:\n {problem_data['input_template']}\n",
            f"The output of the function must follow the template:\n {problem_data['output_template']}\n",
            "Your task is to implement a function 'solve' with a unique argument 'data' as input and returning a solution to the problem.",
            "Write the complete, executable and well indented code of the 'solve' function, including necessary imports.",
            "Status codes are: 'OPTIMAL' for a proven best feasible solution, 'INFEASIBLE' when no feasible solution is found.",
            f"Use OutputFlag = 0 and a TimeLimit of 5 seconds for the optimization. Do not include example usage.",
            ("" if structured_output else f"The code must be enclosed between {open_tag}BEGIN_CODE{close_tag} and {open_tag}END_CODE{close_tag} tags."),
        ])

        answers = llm.batch([query]*n_new)

        for answer in answers:
            if structured_output:
                solvers.append({
                    "code": answer['code'],
                    "self_evaluations": [],
                    "property_list": []
                })
            else:
                try:
                    solvers.append(
                        {
                            "code": parse(answer.content, "CODE")[0],
                            "self_evaluations": [],
                            "property_list": []
                        }
                    )
                except Exception as e:
                    pass
            with open(solvers_file, "wb") as file:
                pickle.dump(solvers, file)

        with open(solvers_file, "wb") as file:
            pickle.dump(solvers, file)

        n_valid = len(solvers)
        n_new = n_solvers - n_valid

    return solvers






##### Instances #####

def generate_instance(problem_data, instances_file, model, n_instances):
    with open(instances_file, "rb") as file:
        instances = pickle.load(file)

    llm_data = ChatOpenAI(model=model, temperature=0.7)
    schema = {
        "title": "DataGenerator",
        "description": "Structured response with complete code of the 'generate_input' function",
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The complete python code for the 'generate_input' function",
            },
        },
        "required": ["code"]
    }
    llm_data = llm_data.with_structured_output(schema)

    diversity_segments = [
        "If possible, the data should be an infeasible instance for the above problem.",
        "The data should be a clearly feasible instance for the above problem in that it admits a simple feasible solution.",
        "The data should results in optimal solutions to the above problem having tight constraints.",
        "The data should be randomized.",
        "The data should be randomized with hyperparameters that will make the solution very likely feasible.",
        "The data should be randomized with hyperparameters that will make the solution very likely infeasible.",
    ]

    while len(instances) < n_instances:
        print(f"# instances: {len(instances)} --> {n_instances}")
        queries = []
        for i in range(len(instances), n_instances):
            seed = ''
            for _ in range(100):
                seed += str(np.random.randint(10))
            query = "\n".join([
                "You are a code-generation agent expert in Python. Numpy is the only library you can use.",
                f"Consider the following problem: {problem_data['problem_description']}",
                f"Here is the template for the problem input: {problem_data['input_template']}",
                "Your task is to implement a function 'generate_input' with no argument and returning a input following the input template.",
                diversity_segments[i % len(diversity_segments)],
                "Write the complete, executable and well indented code of the 'generate_input' function, including necessary imports.",
                "Do not use any NumPy types or objects (e.g., np.array, np.int64). Only use basic built-in Python types: int, float, str, bool, list, dict, None.",
                f"Take inspiration from the following values: {seed}",
            ])
            queries.append(query)

        answers = llm_data.batch(queries)

        for answer in answers:
            try:
                generate_input = str_to_var(answer['code'], 'generate_input')
                data = generate_input()
                instances.append(data)
                with open(instances_file, "wb") as file:
                    pickle.dump(instances, file)
            except Exception as e:
                pass

    return instances





##### Tests #####
def generate_validity_tests(problem_data, validity_tests_file, model, n_tests):
    if os.path.exists(validity_tests_file):
        with open(validity_tests_file, "rb") as file:
            validity_tests = pickle.load(file)
    else:
        validity_tests = []

    n_new = n_tests - len(validity_tests)
    n_valid = len(validity_tests)

    while n_valid < n_tests:
        print(f"# validity tests: {len(validity_tests)} --> {n_tests}")

        structured_output = not (model in ['o3', 'o3-mini'])
        if structured_output:
            llm = ChatOpenAI(model=model, temperature=0.7)
            schema = {
                "title": "CodeGenerator",
                "description": "Structured response with complete code",
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The complete python code for the 'test' function",
                    },
                },
                "required": ["code"]
            }
            llm = llm.with_structured_output(schema)
        else:
            llm = ChatOpenAI(model=model)

        query = "\n".join([
            "You are a code-generation agent expert in Python. numpy, gurobipy, pandas, shapely, and networkx are the only library you can use.",
            f"Here is the problem description:\n {problem_data['problem_description']}\n",
            f"Here is a template for the input data of the problem:\n {problem_data['input_template']}\n",
            f"Here is the solution template of the problem:\n {problem_data['output_template']}\n",
            "For every concrete instance 'data' that follows the input template, there is a corresponding 'solution' object — produced by some solver — that follows the solution template and is supposed to solve that instance.",
            f"Your task is to implement a function test(data, solution) -> bool that returns True if and only if all of the following hold:"
            "  1. The solution is feasible (it satisfies every problem constraint)."
            "  2. The reported objective value matches the cost you compute (within a small numerical tolerance)."
            "  3. All solution fields are internally coherent.\n"
            f"Write the complete, executable and well indented code implementing the 'test' function.",
            f"The code must contain necessary imports. Do not include example usage.",
            ("" if structured_output else f"The code must be enclosed between {open_tag}BEGIN_CODE{close_tag} and {open_tag}END_CODE{close_tag} tags.")
        ])

        queries = [query] * n_new
        answers = llm.batch(queries)

        for answer in answers:
            if structured_output:
                validity_tests.append(answer["code"])
            else:
                validity_tests.append(parse(answer.content, 'CODE')[0])
            with open(validity_tests_file, "wb") as file:
                pickle.dump(validity_tests, file)

        n_valid = len(validity_tests)
        n_new = n_tests - n_valid

    return validity_tests


##### Execution #####

def compute_outputs(solvers_file, instances_file, recompute = True):
    with open(solvers_file, "rb") as file:
        solvers = pickle.load(file)
    with open(instances_file, "rb") as file:    
        instances = pickle.load(file)

    def run_code(code, instance):
        output = execute_code(
            code, 
            'solve', 
            [instance]
        )
        if output[0] != 0:
            return None
        return output[1]

    for solver in tqdm(solvers, desc="Computing outputs of solver-instances pairs"):
        if (not recompute) and ('outputs' in solver and len(solver['outputs']) == len(instances)):
            continue
        solver["outputs"] = []
        solver["interpretable_outputs"] = []
        has_timed_out = False
        for instance in instances:
            if has_timed_out:
                solver["outputs"].append(None)
                solver["interpretable_outputs"].append(False)
                continue
            try:
                output = func_timeout(8, run_code, args=(solver['code'], instance))
            except FunctionTimedOut:
                output = None
                has_timed_out = True
            except Exception as e:
                output = None
            solver["outputs"].append(output)
            if output and "status" in output:
                solver["interpretable_outputs"].append(True)
            else:
                solver["interpretable_outputs"].append(False)

    with open(solvers_file, "wb") as file:
        pickle.dump(solvers, file)

    return solvers



def run_validity_tests(solvers_file, tests_file, instances_file, time_limit=10):

    def run_test(test, instance, output):
        with suppress_output():
            output = execute_code(
                test, 
                'test', 
                [ 
                    instance,
                    output
                ]
            )
        if output[0] != 0:
            return False
        return output[1]
    

    with open(solvers_file, "rb") as file:
        solvers = pickle.load(file)
    with open(tests_file, "rb") as file:
        tests = pickle.load(file)
    with open(instances_file, "rb") as file:    
        instances = pickle.load(file)

    for solver in tqdm(solvers, desc="Running validity tests"):
        solver["validity_tests"] = []
        for test in tests:
            results = []
            for instance, output in zip(instances, solver["outputs"]):
                try:
                    passed = func_timeout(time_limit, run_test, args=(test, instance, output))
                except FunctionTimedOut:
                    passed = None
                except Exception as e:
                    passed = None
                results.append(passed)
            solver["validity_tests"].append(results)
    with open(solvers_file, "wb") as file:
        pickle.dump(solvers, file)

    return solvers
