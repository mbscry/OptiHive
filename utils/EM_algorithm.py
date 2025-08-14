import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln


def build_EM_df(solvers, S, I, T):
    rows = []
    compact_to_original_id = {"solvers": {}, "instances": {}, "tests": {}}
    compact_solver_id = 0
    for s_id, solver in enumerate(solvers):
        if S[s_id] == False: continue
        compact_to_original_id["solvers"][compact_solver_id] = s_id
        tests_obj = solver["validity_tests"]
        compact_test_id = 0
        for t_id, results in enumerate(tests_obj):
            if T[t_id] == False: continue
            compact_to_original_id["tests"][compact_test_id] = t_id
            compact_instance_id = 0
            for i_id, outcome in enumerate(results):
                if I[i_id] == False: continue
                compact_to_original_id["instances"][compact_instance_id] = i_id
                if solvers[s_id]['outputs'][i_id]['status'] != 'INFEASIBLE':
                    rows.append(
                        dict(
                            solver=compact_solver_id,
                            instance=compact_instance_id,
                            test=compact_test_id,
                            result=int(bool(outcome)),
                        )
                    )
                compact_instance_id += 1
            compact_test_id += 1
        compact_solver_id += 1

    df = pd.DataFrame(rows, columns=["solver", "instance", "test", "result"])

    return df, compact_to_original_id


def _log_beta_binom(k, n, a, b):
    log_comb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return log_comb + betaln(k + a, n - k + b) - betaln(a, b)


def _map_update(successes, trials, prior_a, prior_b, eps=1e-9):
    estimate = (successes + prior_a - 1) / (trials + prior_a + prior_b - 2 + eps)
    return np.clip(estimate, eps, 1 - eps)


def EM_algorithm(df, priors, n_solvers, n_instances, n_tests, max_iter=100, tol=1e-6, eps=1e-9):

    R = np.zeros((n_solvers, n_instances), dtype=np.uint8)   
    C = np.zeros((n_solvers, n_instances), dtype=np.int32)
    N = np.zeros((n_solvers, n_instances), dtype=np.int32)

    for s, i, passed in df[["solver", "instance", "result"]].values:
        R[s, i] = 1
        C[s, i] += passed
        N[s, i] += 1

    lambd = 0.5
    alpha_s = 0.1 * np.ones(n_solvers)
    beta_s = 0.1 * np.ones(n_solvers)
    gamma_s = np.clip(df.groupby("solver")["result"].mean().reindex(range(n_solvers)).fillna(0.1).values, 1e-3, 1-1e-3)
    fsi_hat = 0.5 * np.ones((n_solvers, n_instances))
    rho0, rho1 = 0.5, 0.5
    p0, p1 = 0.1, 0.9

    for itr in range(max_iter):
        # E-STEP
        # w1 = fsi_hat * R
        # w0 = (1 - fsi_hat) * R
        # p0 = (w0 * (C / (N + eps))).sum() / (w0.sum() + eps)

        # p1 = _map_update((w1 * (C / (N + eps))).sum(), w1.sum(), *priors["ab_bar"])

        a0, b0 = p0 * (1 / rho0 - 1), (1 - p0) * (1 / rho0 - 1)
        a1, b1 = p1 * (1 / rho1 - 1), (1 - p1) * (1 / rho1 - 1)

        logL0 = _log_beta_binom(C, N, a0, b0)
        logL1 = _log_beta_binom(C, N, a1, b1)

        log_like1 = (1 - R) * np.log(beta_s + eps)[:, None]
        log_like0 = (1 - R) * np.log(1 - alpha_s + eps)[:, None]

        log_mix = np.logaddexp(
            np.log(gamma_s + eps)[:, None] + logL1,
            np.log(1 - gamma_s + eps)[:, None] + logL0,
        )

        log_like1 += R * (np.log(1 - beta_s + eps)[:, None] + log_mix)
        log_like0 += R * (np.log(alpha_s + eps)[:, None] + logL0)

        logLambda1 = log_like1.sum(0)
        logLambda0 = log_like0.sum(0)

        log_fi_num = np.log(lambd + eps) + logLambda1
        log_fi_den = np.logaddexp(log_fi_num, np.log(1 - lambd + eps) + logLambda0)
        fi_hat = np.exp(log_fi_num - log_fi_den)

        fsi_hat = np.zeros_like(C, dtype=float)
        s_idx, i_idx = np.where(R)
        num = np.log(gamma_s[s_idx] + eps) + logL1[s_idx, i_idx]
        den = np.logaddexp(num, np.log(1 - gamma_s[s_idx] + eps) + logL0[s_idx, i_idx])
        fsi_hat[s_idx, i_idx] = fi_hat[i_idx] * np.exp(num - den)

        # M-STEP
        lambd = _map_update(fi_hat.sum(), n_instances, *priors["lambda"])
        alpha_s = _map_update(((1 - fi_hat)[None, :] * R).sum(1), (1 - fi_hat).sum(), *priors["alpha_s"])
        beta_s  = _map_update((fi_hat[None, :] * (1 - R)).sum(1), fi_hat.sum(), *priors["beta_s"])
        gamma_s = _map_update((fsi_hat * R).sum(1), (fi_hat[None, :] * R).sum(1), *priors["gamma_s"])

        p0 = _map_update(((1 - fsi_hat) * R * C).sum(), ((1 - fsi_hat) * R * N).sum(), *priors["p0"])
        p1 = 1 - _map_update((fsi_hat * R * (N - C)).sum(), (fsi_hat * R * N).sum(), priors["p1"][1], priors["p1"][0])

        def _update_rho(weight):
            mask = (weight > 0)
            w = weight[mask]
            c = C[mask]
            n = N[mask]
            n_bar = (w * n).sum() / (w.sum() + eps)
            p_hat = (w * c / n).sum() / (w.sum() + eps)
            var_hat = (w * (c / n - p_hat) ** 2).sum() / (w.sum() + eps)
            rho = (n_bar * var_hat / (p_hat * (1 - p_hat) + eps) - 1) / (n_bar - 1 + eps)
            return np.clip(rho, 1e-5, 1 - 1e-5)

        rho1 = _update_rho(fsi_hat * R)
        rho0 = _update_rho((1 - fsi_hat) * R)

        # Convergence check
        if itr:
            delta = max(
                abs(np.log(lambd) - np.log(prev_lambda)),
                np.max(abs(np.log(alpha_s) - np.log(prev_alpha_s))),
                np.max(abs(np.log(beta_s)  - np.log(prev_beta_s))),
                np.max(abs(np.log(gamma_s) - np.log(prev_gamma_s))),
                abs(p0 - prev_p0),
                abs(p1 - prev_p1),
                abs(rho1 - prev_rho1),
                abs(rho0 - prev_rho0),
            )
            if delta < tol:
                break

        prev_lambda, prev_alpha_s, prev_beta_s, prev_gamma_s = (
            lambd, alpha_s.copy(), beta_s.copy(), gamma_s.copy()
        )
        prev_p0, prev_p1 = p0, p1
        prev_rho1, prev_rho0 = rho1, rho0

    return {
        "lambda": lambd,
        "alpha_s": alpha_s,
        "beta_s": beta_s,
        "gamma_s": gamma_s,
        "a0": a0, "b0": b0,
        "a1": a1, "b1": b1,
        "p0": p0, "p1": p1,
        "rho0": rho0, "rho1": rho1,
        "fi": fi_hat,
        "fsi": fsi_hat,
        "iterations": itr + 1,
    }