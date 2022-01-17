from ast import Num
import torch
from torch.autograd import Function
import torch.nn as nn
import ot


def pot_sinkhorn(M, a, b, epsilon, **solver_options):
    return ot.bregman.sinkhorn_log( #ot.sinkhorn(
        a,
        b,
        M,
        epsilon,
        stopThr=1e-02,
        **solver_options
    )


def sinkhorn(M, a, b, epsilon, n_iter):
    """Basic sinkhorn algorithm.

    Solves the regularized OT problem:
        max <M, P> + \epsilon H[P] s.t. \sum_j P_ij = a_i and \sum_i P_ij = b_j
    with entropy H[P] = - \sum_ij P_ij [log(P_ij) - 1]

    Args:
        M (torch.Tensor): affinity matrix
        a (torch.Tensor): user capacities
        b (torch.Tensor): item capacities
        epsilon (float): regularization parameter
        n_iter (int): number of iterations

    Returns:
        P (torch.Tensor): coupling matrix
    """
    v = torch.ones_like(b)
    K = torch.exp(M / epsilon)

    for _ in range(n_iter):
        u = a / torch.matmul(K, v)
        v = b / torch.matmul(torch.transpose(K, 1, 0), u)

    uv = torch.outer(u, v)
    P = K * uv

    return P


class SinkhornLossFunc(Function):
    @staticmethod
    def forward(ctx, M, target, a, b, epsilon, solver, solver_options):
        P = solver(M.detach(), a, b, epsilon, **solver_options)
        cross_entropy = - (target * P.log()).sum()
        delta_P = (P - target) / epsilon
        ctx.save_for_backward(delta_P)

        return cross_entropy

    @staticmethod
    def backward(ctx, grad_output):
        delta_P, = ctx.saved_tensors
        grad_M = delta_P * grad_output

        return grad_M, None, None, None, None, None, None


class SinkhornLoss(nn.Module):
    """Sinkhorn loss.

    Computes loss = H[target, P(M)] where P(M) is the solution of the
    regularized OT problem with affinity matrix M.

    Args:
        a (torch.Tensor): user capacities
        b (torch.Tensor): item capacities
        epsilon (float): regularization parameter
        solver (function): OT solver
        solver_kwargs (int): options to pass to the solver
    """
    def __init__(self, a, b, epsilon, solver, **solver_options):
        super().__init__()
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.solver = solver
        self.solver_options = solver_options

    def forward(self, M, target):
        return SinkhornLossFunc.apply(
            M,
            target,
            self.a,
            self.b,
            self.epsilon,
            self.solver,
            self.solver_options
        )

    def extra_repr(self):
        return (
            f"a={self.a},\nb={self.b},\n"
            f"epsilon={self.epsilon:.2e}, solver={self.solver}, solver_options={self.solver_options}"
        )


class SinkhornValueFunc(Function):
    @staticmethod
    def forward(ctx, M, a, b, epsilon, solver, solver_options):
        P = solver(
            M,
            a,
            b,
            epsilon,
            **solver_options
        )
        ctx.save_for_backward(P)
        # clamping log(P) to -100 to avoid 0 log(0) = nan
        log_P = P.log().clamp(min=-100)
        H = (P * (1 - log_P)).sum()
        value_OT = (P*M).sum() + epsilon*H

        return (P*M).sum() #value_OT

    @staticmethod
    def backward(ctx, grad_output):
        P, = ctx.saved_tensors
        grad_M = P * grad_output

        return grad_M, None, None, None, None, None


class SinkhornValue(nn.Module):
    """Sinkhorn value.

    Returns optimal value for the regularized OT problem:
        L(M) = max <M, P> + \epsilon H[P] s.t. \sum_j P_ij = a_i and \sum_i P_ij = b_j
    with entropy H[P] = - \sum_ij P_ij [log(P_ij) - 1]

    Args:
        a (torch.Tensor): user capacities
        b (torch.Tensor): item capacities
        epsilon (float): regularization parameter
        solver (function): OT solver
        solver_kwargs (int): options to pass to the solver
    """
    def __init__(self, a, b, epsilon, solver, **solver_options):
        super().__init__()
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.solver = solver
        self.solver_options = solver_options

    def forward(self, M):
        return SinkhornValueFunc.apply(
            M,
            self.a,
            self.b,
            self.epsilon,
            self.solver,
            self.solver_options
        )

    def extra_repr(self):
        return (
            f"a={self.a},\nb={self.b},\n"
            f"epsilon={self.epsilon:.2e}, solver={self.solver}, solver_options={self.solver_options}"
        )
