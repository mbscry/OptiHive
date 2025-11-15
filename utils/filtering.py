from gurobipy import Model, GRB, quicksum

def maximal_component_selection(solvers):
    m = Model("selection")
    m.Params.OutputFlag = 0

    ns = len(solvers)
    ni = len(solvers[0]["interpretable_outputs"])
    nt = len(solvers[0]["validity_tests"])

    x_s = {i: m.addVar(vtype=GRB.BINARY, name=f"s_{i}") for i in range(ns)}
    x_i = {j: m.addVar(vtype=GRB.BINARY, name=f"i_{j}") for j in range(ni)}
    x_t = {l: m.addVar(vtype=GRB.BINARY, name=f"f_{l}") for l in range(nt)}

    for i in range(ns):
        solver = solvers[i]
        for j in range(ni):
            passed = solver["interpretable_outputs"][j]
            if not passed:
                m.addConstr(x_s[i] + x_i[j] <= 1, f"c_{i}_{j}")
            else:
                for l in range(nt):
                    if solver["validity_tests"][l][j] is None:
                        m.addConstr(x_s[i] + x_i[j] + x_t[l] <= 2, f"c_{i}_{j}_{l}")

    m.addConstr(quicksum(x_s.values()) >= 1, "c_min_solvers")
    m.addConstr(quicksum(x_i.values()) >= 1, "c_min_outputs")
    m.addConstr(quicksum(x_t.values()) >= 1, "c_min_validity_tests")
    m.setObjective(quicksum(x_s.values()) + quicksum(x_i.values()) + quicksum(x_t.values()), GRB.MAXIMIZE)
    m.optimize()

    S, I, T = [], [], []
    for i in x_s:
        S.append(x_s[i].X > 0.5)
    for i in x_i:
        I.append(x_i[i].X > 0.5)
    for i in x_t:
        T.append(x_t[i].X > 0.5)

    return S, I, T