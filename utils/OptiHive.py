import os
import pickle
from tqdm import tqdm
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

from utils.generation import generate_solvers, generate_instance, generate_validity_tests, execute_code, compute_outputs, run_validity_tests
from utils.EM_algorithm import EM_algorithm, build_EM_df
from utils.filtering import maximal_component_selection

class OptiHive:
    
    def __init__(self, problem_folder, solver_model = "gpt-4.1-mini", instance_model = "gpt-4.1-mini", test_model = "gpt-4.1-mini", SOLVERS_BATCH_SIZE = 1):
        self.problem_folder = problem_folder
        self.solver_model = solver_model
        self.instance_model = instance_model
        self.test_model = test_model
        self.SOLVERS_BATCH_SIZE = SOLVERS_BATCH_SIZE

        with open(f"{self.problem_folder}/problem_data.pkl", "rb") as file:
            self.problem_data = pickle.load(file)



    def generate_components(self, n_solvers, n_instances, n_tests):
        print("Generating components...")
        solvers_file = f"{self.problem_folder}/solvers_{self.solver_model}.pkl"
        instances_file = f"{self.problem_folder}/instances_{self.instance_model}.pkl"
        validity_tests_file = f"{self.problem_folder}/validity_tests_{self.test_model}.pkl"
        if not os.path.exists(solvers_file):
            os.makedirs(os.path.dirname(solvers_file), exist_ok=True)

        # Load existing components
        if os.path.exists(solvers_file):
            with open(solvers_file, "rb") as file:
                solvers = pickle.load(file)
        else:
            solvers = []
            with open(solvers_file, "wb") as file:
                pickle.dump(solvers, file)

        if os.path.exists(instances_file):
            with open(instances_file, "rb") as file:
                instances = pickle.load(file)
        else:
            instances = []
            with open(instances_file, "wb") as file:
                pickle.dump(instances, file)

        if os.path.exists(validity_tests_file):
            with open(validity_tests_file, "rb") as file:
                validity_tests = pickle.load(file)
        else:
            validity_tests = []
            with open(validity_tests_file, "wb") as file:
                pickle.dump(validity_tests, file)

        # Generate solvers by batches of SOLVERS_BATCH_SIZE
        while len(solvers) < n_solvers:
            solvers = generate_solvers(
                solvers_file,
                self.problem_data,
                model = self.solver_model,
                n_solvers = min(n_solvers, len(solvers) + self.SOLVERS_BATCH_SIZE),
            )

        # Generate instances
        instances = generate_instance(
            self.problem_data,
            instances_file,
            self.instance_model,
            n_instances
        )

        # Generate validity tests
        validity_tests = generate_validity_tests(
            self.problem_data,
            validity_tests_file,
            self.test_model,
            n_tests
        )
        print("Components loaded/generated successfully.")



    def evaluate(self, verbose = False):
        solvers_file = f"{self.problem_folder}/solvers_{self.solver_model}.pkl"
        instances_file = f"{self.problem_folder}/instances_{self.instance_model}.pkl"
        validity_tests_file = f"{self.problem_folder}/validity_tests_{self.test_model}.pkl"

        with open(solvers_file, "rb") as file:
            solvers = pickle.load(file)

        for solver in tqdm(solvers, desc="Evaluating solvers"):
            is_correct = True
            is_feasible = True
            executable = False
            interpretable = False
            crash = False
            for instance in self.problem_data["validation_set"]:
                data = instance['data']
                try:
                    with np.errstate(invalid="ignore"):
                        r = func_timeout(10, execute_code, args=(solver["code"], 'solve', [data]))
                        if r[0] == 0:
                            executable = True
                            if "status" in r[1]:
                                interpretable = True
                            else:
                                is_feasible = False
                        else:
                            is_correct = False
                            is_feasible = False
                            crash = True
                        r = r[1]
                        if r["status"] == "INFEASIBLE":
                            if instance['status'] != "INFEASIBLE":
                                is_correct = False
                        else:
                            if instance['status']=="INFEASIBLE" or abs(r["objective_value"] - instance['objective_value']) / instance['objective_value'] > 1e-3:
                                is_correct = False
                            if self.problem_data['evaluate_feasibility'](data, r) != True:
                                is_feasible = False
                                is_correct = False
                except FunctionTimedOut:
                    is_correct = False
                    is_feasible = False
                    break
                except Exception as e:
                    is_correct = False
                    is_feasible = False
                    crash = True
                    break

            solver["correct"] = is_correct
            solver["executable"] = executable
            solver["interpretable"] = interpretable
            solver["crash"] = crash
            solver["feasible"] = is_feasible

            with open(solvers_file, "wb") as file:
                pickle.dump(solvers, file)

        if verbose:
            n_solvers = len(solvers)
            print('-'*30)
            print("# solvers =", n_solvers)
            print("% correct =", sum(solver['correct'] for solver in solvers) / n_solvers * 100)
            print("% feasible =", sum(solver['feasible'] for solver in solvers) / n_solvers * 100)
            print("% executable =", sum(solver['executable'] for solver in solvers) / n_solvers * 100)
            print("% interpretable =", sum(solver['interpretable'] for solver in solvers) / n_solvers * 100)
            print("% crashed =", sum(solver['crash'] for solver in solvers) / n_solvers * 100)
            print('-'*30)

        solvers = compute_outputs(solvers_file, instances_file)
        solvers = run_validity_tests(solvers_file, validity_tests_file, instances_file, time_limit=1)

        return solvers
    


    def run_selection(self, n_solvers, n_instances, n_validity_tests):

        solvers_file = f"{self.problem_folder}/solvers_{self.solver_model}.pkl" 
        with open(solvers_file, "rb") as file:
            solvers = pickle.load(file)

        # Sample desired number of solvers, instances, and validity tests
        indexes_solvers = np.random.choice(len(solvers), size=n_solvers, replace=True).tolist()
        reduced_to_original_id  = {j: idx for j, idx in enumerate(indexes_solvers)}
        indexes_instances = (np.random.choice(len(solvers[0]["outputs"]), size=n_instances, replace=True).tolist())
        indexes_validity_tests = (np.random.choice(len(solvers[0]["validity_tests"]), size=n_validity_tests, replace=True).tolist())

        solver_samples = []
        optimal_sampled = 0
        for i in indexes_solvers:
            solver = {}
            solver["outputs"] = [solvers[i]["outputs"][j] for j in indexes_instances]
            solver["interpretable_outputs"] = [solvers[i]["interpretable_outputs"][j] for j in indexes_instances]
            solver["validity_tests"] = [[solvers[i]["validity_tests"][l][j] for j in indexes_instances] for l in indexes_validity_tests]
            solver['correct'] = solvers[i]["correct"]
            solver['executable'] = solvers[i]["executable"]
            solver_samples.append(solver)
            optimal_sampled += solvers[i]["correct"]

        try:
            # Filter components
            S, I, T = maximal_component_selection(solver_samples)
            # Build the DataFrame for EM algorithm containing interpretable components
            df, compact_to_reduced_id = build_EM_df(solver_samples, S, I, T)
        except Exception as e:
            return [[i, -1] for i in indexes_solvers], None, optimal_sampled
        if df.empty:
            return [[i, -1] for i in indexes_solvers], None, optimal_sampled

        # Run EM algorithm
        priors = {
            "lambda": (1, 1),
            "alpha_s": (1, 1),
            "beta_s": (1, 1),
            "gamma_s": (1, 1),
            "p0": (1, 1),
            "p1": (20, 1),
        }
        nS = len(compact_to_reduced_id["solvers"])
        nI = len(compact_to_reduced_id["instances"])
        nT = len(compact_to_reduced_id["tests"])
        results_em = EM_algorithm(df, priors, nS, nI, nT, max_iter=100)

        # Compute scalarized objective
        parameters = {}

        obj_max = 1e3
        for s_id in range(len(results_em['alpha_s'])):
            reduced_id = compact_to_reduced_id["solvers"][s_id]
            for j in range(results_em['fsi'].shape[1]):
                o = solver_samples[reduced_id]["outputs"][compact_to_reduced_id["instances"][j]]
                if o["status"] == "OPTIMAL":
                    obj_max = max(abs(obj_max), o["objective_value"])

        P_inf = 10 * obj_max
        P_miss = 10 * obj_max
        lambd = results_em['lambda']

        for s_id in range(len(results_em['alpha_s'])):
            alpha = results_em['alpha_s'][s_id]
            beta = results_em['beta_s'][s_id]
            gamma = results_em['gamma_s'][s_id]

            reduced_id = compact_to_reduced_id["solvers"][s_id]
            solver = solver_samples[reduced_id]
            num = 0
            den = 0
            for j in range(results_em['fsi'].shape[1]):
                o = solver["outputs"][compact_to_reduced_id["instances"][j]]
                if o["status"] == "OPTIMAL":
                    num += o["objective_value"] * results_em['fsi'][s_id, j]
                    den += results_em['fsi'][s_id, j]

            conditional_objective = num / den * (-1,1)[self.problem_data['minimization']] if den > 1e-5 else (obj_max)

            Vs = ((1-lambd) * alpha * P_inf + lambd * (beta * P_miss + (1-beta) * (gamma * conditional_objective + (1-gamma) * P_inf)))
            parameters[reduced_to_original_id[reduced_id]] = {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "conditional_objective": conditional_objective,
                "Vs": Vs
            }
        
        ranking = sorted(parameters.items(), key=lambda item: item[1]["Vs"])

        return ranking, results_em, optimal_sampled