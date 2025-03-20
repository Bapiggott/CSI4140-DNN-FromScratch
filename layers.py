import torch

class FullyConnectedLayer:
    """
    Manually implemented fully connected (linear) layer.
    """
    def __init__(self, in_features, out_features, init_scale=0.01, device=torch.device("cpu")):
        # TODO: Initialize self.W, self.b, self.dW, self.db
        self.W = torch.randn(in_features, out_features, device=device) * init_scale
        self.b = torch.zeros(out_features, device=device)

        self.dW = torch.zeros_like(self.W)
        self.db = torch.zeros_like(self.b)

        self.x = None
        #pass

    def forward(self, x):
        """
        x shape: (batch_size, in_features)
        Returns: (batch_size, out_features)
        """
        self.x = x

        z = x @ self.W.t() + self.b
        return z

        #pass

    def backward(self, grad_output):
        """
        grad_output shape: (batch_size, out_features)
        Returns: grad_input shape: (batch_size, in_features)
        """

        batch_size = grad_output.shape[0]

        self.dW = self.x.T @ grad_output
        self.db = grad_output.sum(dim=0)

        pass


class Conv2DLayer:
    """
    Manually implemented 2D convolution layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device=torch.device("cpu")):
        # TODO: Initialize self.W, self.b, self.dW, self.db
        #to device thing
        pass

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, height, width)
        Returns: (batch_size, out_channels, out_height, out_width)
        """
        pass

    def backward(self, grad_output):
        """
        grad_output shape: (batch_size, out_channels, out_height, out_width)
        Returns: grad_input shape: (batch_size, in_channels, height, width)
        """
        pass


class ReLU:
    """
    Manually implemented ReLU activation.
    """
    def __init__(self):
        pass

    def forward(self, x):
        """
        x shape: same as input
        Returns: element-wise max(x, 0)
        """
        pass

    def backward(self, grad_output):
        """
        Zeros out gradient where the original x was < 0
        """
        pass


class Dropout:
    """
    Manually implemented dropout layer.
    """
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward(self, x, train=True):
        """
        If train=True, zero out fraction p of inputs and scale.
        If train=False, pass x as-is.
        """
        pass

    def backward(self, grad_output):
        pass


class SoftmaxCrossEntropy:
    """
    Combined Softmax + CrossEntropy for stability.
    """
    def __init__(self):
        pass

    def forward(self, logits, labels):
        """
        logits: (batch_size, num_classes)
        labels: (batch_size) with class indices
        Returns: scalar loss
        """
        pass

    def backward(self):
        """
        Returns gradient wrt logits, shape (batch_size, num_classes).
        """
        pass


class ResidualBlock:
    """
    Example of a residual (RAG-style) block with skip connections.
    Uses multiple Conv2D + ReLU layers internally, then sums skip.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        - main path: Conv -> ReLU -> Conv
        - skip path: either identity or a 1x1 conv if shape changes
        """
        # Main path sub-layers
        self.conv1 = Conv2DLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.relu1 = ReLU()
        self.conv2 = Conv2DLayer(out_channels, out_channels, kernel_size, stride=1, padding=1)

        # For the skip path:
        self.skip_conv = None
        if (in_channels != out_channels) or (stride != 1):
            # E.g., 1x1 conv if mismatch in channels/size
            self.skip_conv = Conv2DLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        # ReLU after addition
        self.relu2 = ReLU()

        # Will store for backward pass
        self.x = None
        self.out_main = None
        self.out_skip = None

    def forward(self, x):
        """
        Forward pass: main path, plus skip path, then add them, then final ReLU.
        """
        pass

    def backward(self, grad_output):
        """
        Backprop through main path, skip path, then sum gradient.
        """
        pass
