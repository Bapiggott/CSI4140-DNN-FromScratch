from layers import (
    FullyConnectedLayer,
    Conv2DLayer,
    ReLU,
    Dropout,
    SoftmaxCrossEntropy,
    ResidualBlock
    # etc.
)

class Model:
    """
    Basic sequential model that can contain normal layers or residual blocks.
    """
    def __init__(self, layers):
        """
        layers: a list of layer instances (Conv2DLayer, ReLU, ResidualBlock, etc.)
        """
        self.layers = layers

    def forward(self, x, train=True):
        for layer in self.layers:
            # If you need to pass train mode (for Dropout), do so
            # otherwise just call forward(x).
            # e.g.:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train=train)
            else:
                x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self):
        """
        Collects (param, grad) pairs from each layer.
        For ResidualBlock, also gather from its sub-layers.
        """
        for layer in self.layers:
            # Normal layer
            if hasattr(layer, 'W'):
                yield (layer.W, layer.dW)
            if hasattr(layer, 'b'):
                yield (layer.b, layer.db)

            # If it's a ResidualBlock, gather from sub-layers
            if isinstance(layer, ResidualBlock):
                # For example:
                if hasattr(layer.conv1, 'W'):
                    yield (layer.conv1.W, layer.conv1.dW)
                if hasattr(layer.conv1, 'b'):
                    yield (layer.conv1.b, layer.conv1.db)
                # do the same for layer.conv2
                if hasattr(layer.conv2, 'W'):
                    yield (layer.conv2.W, layer.conv2.dW)
                if hasattr(layer.conv2, 'b'):
                    yield (layer.conv2.b, layer.conv2.db)
                # if skip_conv exists:
                if layer.skip_conv is not None:
                    if hasattr(layer.skip_conv, 'W'):
                        yield (layer.skip_conv.W, layer.skip_conv.dW)
                    if hasattr(layer.skip_conv, 'b'):
                        yield (layer.skip_conv.b, layer.skip_conv.db)
                # also check if there's anything else to yield
