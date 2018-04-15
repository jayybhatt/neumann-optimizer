import math
import torch
from torch.optim.optimizer import Optimizer

class Neumann(Optimizer):
    """
    Documentation about the algorithm
    """

    def __init__(self, params, lr=1e-3,eps = 1e-8):
        defaults = dict(lr=lr, eps=eps,
                    )

        super(Neumann, self).__init__(params, defaults)


    def step():
        """
        Performs a single optimization step.
        
        """

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data


        