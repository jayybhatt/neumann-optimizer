import math
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
import numpy as np
# import pdb; pdb.set_trace()

class Neumann(Optimizer):
    """
    Documentation about the algorithm
    """

    def __init__(self, params , lr=1e-3,eps = 1e-8, alpha = 1e-7, beta = 1e-5, gamma = 0.9, momentum = 1, sgd_steps = 5, K = 10 ):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 1 >= momentum:
            raise ValueError("Invalid momentum value: {}".format(eps))


        self.iter = 0
        self.sgd = SGD(params, lr=lr, momentum=0.9)

        param_count = np.sum([np.prod(p.size()) for p in params]) # got from MNIST-GAN

        defaults = dict(lr=lr, eps=eps, alpha=alpha,
                    beta=beta*param_count, gamma=gamma,
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

            alpha = group['alpha']
            beta = group['beta']
            gamma = group['gamma']
            K = group['K']
            momentum = group['momentum']
            mu = momentum*(1 - (1/(1+self.iter)))
            eta = group['lr']/self.iter ## update with time


            if self.iter <= sgd_steps:
                self.sgd.step()
                return

            if self.iter == 8:
                print("here")

            if mu >= 0.9:
                mu = 0.9
            elif mu <= 0.5:
                mu = 0.5

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


                ## Reset neumann iterate
                if self.iter%K == 0:
                    state['m'] = grad.mul(-eta)
                    group['K'] = group['K']*2

                ## Compute update d_t
                diff = p.data.sub(state['moving_avg'])
                diff_norm = (p.data.sub(state['moving_avg'])).norm(p=1)
                if np.count_nonzero(diff):
                    state['d'] = grad.add( (( (diff_norm.pow(2)).mul(alpha) ).sub( (diff_norm.pow(-2)).mul(beta) )).mul( diff.div(diff_norm)) )
                else:
                    state['d'] = grad

                ## Update Neumann iterate
                (state['m'].mul_(mu)).sub_( state['d'].mul(eta))

                ## Update Weights
                p.data.add_((state['m'].mul(mu)).sub( state['d'].mul(eta)))

                ## Update Moving Average
                state['moving_avg'] = p.data.add( (state['moving_avg'].sub(p.data)).mul(gamma) )




        # return loss
