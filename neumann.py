import math
import torch
from torch.optim.optimizer import Optimizer
import pdb; pdb.set_trace()

class Neumann(Optimizer):
    """
    Documentation about the algorithm
    """

    def __init__(self, params , lr=1e-3,eps = 1e-8, alpha = 1e-7, beta = 1e-5, gamma = 0.9, momentum = 0.5, sgd_steps = 5, K = 10 ):
        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.9 >= mu:
            raise ValueError("Invalid mu value: {}".format(eps))
        

        num_variables = 2#calculate here
        defaults = dict(lr=lr, eps=eps, alpha=alpha,
                    beta=beta*num_variables, gamma=gamma,
                    sgd_steps=sgd_steps, momentum=momentum, K=K
                    )

        super(Neumann, self).__init__(params, defaults)


    def step(self, closure):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None: #checkout what's the deal with this. present in multiple pytorch optimizers
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data


                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0

                    state['m'] = torch.zeros_like(p.data).float()

                    state['d'] = torch.zeros_like(p.data).float()

                    state['moving_avg'] = torch.zeros_like(p.data).float()

                    state['K'] = 10




                state['step'] += 1





        
        return loss


        