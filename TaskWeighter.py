import cvxpy as cp
import numpy as np

def _safely_solve(prob):
    try:
        prob.solve()
        assert prob.status in ["optimal", "optimal_inaccurate"], "Opt problem is infeasible or unbounded"
    except Exception as e:
        raise RuntimeError("Could not solve for task weights.") from e

# TODO: This is slower than it needs to be because it is not a parameterized optimization problem and therefore has to convert to standard form at each step. This can be improved with proper parameterization.
def _compute_genpcgrad_weighting(M):
    T, _ = M.shape

    lambd = cp.Variable(T)
    M_sqrt = np.linalg.cholesky(M).T

    ml = M @ lambd

    obj = 0.5 * cp.sum_squares(M_sqrt @ (lambd - 1.0/T * np.ones(T)))
    prob = cp.Problem(cp.Minimize(obj), [ml >= 0, cp.sum(lambd) == 1])

    _safely_solve(prob)

    return lambd.value.astype(np.float32)

def compute_task_weightings(proj_type, M):
    if proj_type == "Gen-PCGrad":
        return _compute_genpcgrad_weighting(M)
    else:
        raise ValueError(f"Proj type {proj_type} not recognized")
