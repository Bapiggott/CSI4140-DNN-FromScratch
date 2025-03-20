import torch

def l2_regularization(model, lambda_):
    """
    Sum up the squares of the weights of each layer/block.
    Return 0.5 * lambda_ * sum(...)
    """
    pass

def l2_grad(model, lambda_):
    """
    Add lambda_ * W to each layer's dW (in-place).
    """
    pass
