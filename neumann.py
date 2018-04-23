import math
import torch
from .optimizer import Optimizer
from .sgd import SGD

class Neumann(Optimizer):
    """
    Documentation about the algorithm
    """

    def __init__(self, params , lr=1e-3,eps = 1e-8, alpha = 1e-7, beta = 1e-5, gamma = 0.9, momentum = 0.5, sgd_steps = 5, K = 10 ):
        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.9 >= momentum:
            raise ValueError("Invalid momentum value: {}".format(eps))
        

        self.iter = 0
        self.sgd = SGD(params, lr=lr, momentum=0.9)

        num_variables = 2#calculate here
        defaults = dict(lr=lr, eps=eps, alpha=alpha,
                    beta=beta*num_variables, gamma=gamma,
                    sgd_steps=sgd_steps, momentum=momentum, K=K
                    )

        super(Neumann, self).__init__(params, defaults)


    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.iter += 1


        loss = None
        if closure is not None: #checkout what's the deal with this. present in multiple pytorch optimizers
            loss = closure()

        for group in self.param_groups:

            sgd_steps = group['sgd_steps']

            if self.iter <= sgd_steps:
                self.sgd.step()
                return

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
                    state['moving_avg'] = p.data

                state['step'] += 1

                alpha = group['alpha']
                beta = group['beta']
                gamma = group['gamma']
                K = group['K']
                momentum = group['momentum']
                mu = momentum*(1 - (1/(1+self.iter)))
                eta = group['lr']/self.iter ## update with time

                ## Reset neumann iterate 
                if self.iter%K == 1:
                    state['m'] = grad.mul(-eta)

                ## Compute update d_t
                diff = p.data.sub(state['moving_avg'])
                diff_norm = (p.data.sub(state['moving_avg'])).norm()
                state['d'] = grad.add( (( (diff_norm.pow(2)).mul(alpha) ).sub( (diff_norm.pow(-2)).mul(beta) )).mul( diff.div(diff_norm)) )

                ## Update Neumann iterate
                state['m'] = (state['m'].mul_(mu)).sub_( state['d'].mul(eta))

                ## Update Weights
                p.data.add_((state['m'].mul(mu)).sub( state['d'].mul(eta)))

                ## Update Moving Average
                state['moving_avg'] = p.data.add( (state['moving_avg'].sub(p.data)).mul(gamma) )



        
        # return loss