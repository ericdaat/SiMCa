from ast import Num
import torch
from torch.autograd import Function
import torch.nn as nn
import ot


def pot_sinkhorn(M, a, b, epsilon, **solver_options):
    return ot.bregman.sinkhorn_log(
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
            f"epsilon={self.epsilon:.2e}, solver={self.solver},"
            "solver_options={self.solver_options}"
        )


class SinkhornValueFunc(Function):
    @staticmethod
    def forward(ctx, M, stored_M, a, b, epsilon, solver, solver_options):
        # Run Sinkhorn
        P = solver(
            torch.cat([M, stored_M]),  # Use the queue
            a,
            b,
            epsilon,
            **solver_options
        )
        P = P[:M.shape[0], :]  # Take only current batch

        ctx.save_for_backward(P)
        return (P*M).sum()

    @staticmethod
    def backward(ctx, grad_output):
        P, = ctx.saved_tensors
        grad_M = P * grad_output

        return grad_M, None, None, None, None, None, None


class SinkhornValue(nn.Module):
    """Sinkhorn value.

    Returns optimal value for the regularized OT problem:
        L(M) = max <M, P> + \epsilon H[P] s.t. \sum_j P_ij = a_i and \sum_i P_ij = b_j
    with entropy H[P] = - \sum_ij P_ij [log(P_ij) - 1]

    Args:
        epsilon (float): regularization parameter
        solver (function): OT solver
        solver_kwargs (int): options to pass to the solver
    """
    def __init__(self, epsilon, max_n_batches_in_queue, solver,
                 **solver_options):
        super().__init__()
        # Sinkhorn params
        self.epsilon = epsilon
        self.solver = solver
        self.solver_options = solver_options

        # Queue params
        self.stored_M = torch.Tensor()  # tensor acts as queue
        self.max_n_batches_in_queue = max_n_batches_in_queue

    def forward(self, M):
        batch_size = M.shape[0]

        #################
        # Sinkhorn step #
        #################
        # Compute marginals
        M_concat = torch.cat([M, self.stored_M])
        a = torch.ones(M_concat.shape[0])
        b = torch.ones(M.shape[1]) / (M.shape[0] / M.shape[1])

        # Compute sinkhorn
        loss = SinkhornValueFunc.apply(
            M,
            self.stored_M,
            a,
            b,
            self.epsilon,
            self.solver,
            self.solver_options
        )

        ################
        # Update queue #
        ################
        n_batches_in_queue = self.stored_M.shape[0] / batch_size
        if n_batches_in_queue < self.max_n_batches_in_queue:
            # Append current batch to previous batches
            self.stored_M = M_concat
        else:
            # Roll stored M, older batch comes first, replace it with M
            self.stored_M = torch.roll(self.stored_M, batch_size, 0)
            self.stored_M[:batch_size, :] = M

        return loss

    def extra_repr(self):
        return (
            f"epsilon={self.epsilon:.2e}, solver={self.solver},"
            "solver_options={self.solver_options}"
        )
