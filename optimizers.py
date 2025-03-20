import torch
import math

class SGDWithMomentum:
    """
    Manually implemented SGD with momentum.
    """
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self):
        """
        Update each parameter using momentum.
        """
        pass


class Adam:
    """
    Manually implemented Adam optimizer.
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        """
        Update parameters using Adam with bias correction.
        """
        pass


class CosineLRDecay:
    """
    Adjust optimizer.lr using a cosine decay schedule.
    """
    def __init__(self, optimizer, initial_lr, max_epochs):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs

    def step(self, current_epoch):
        pass


class ExponentialLRDecay:
    """
    Adjust optimizer.lr using an exponential decay schedule.
    """
    def __init__(self, optimizer, initial_lr, decay_rate):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def step(self, current_epoch):
        pass
