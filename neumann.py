from keras.optimizers import Optimizer
from keras.backend import K

class Neumann(Optimizer):
    """Neumann Optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Neumann Optimizer: A Practical Optimization Algorithm for Deep Neural Networks](https://arxiv.org/pdf/1712.03298.pdf)
    """
    def __init__(self, epsilon=None, **kwargs):
        super(Neumann, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
