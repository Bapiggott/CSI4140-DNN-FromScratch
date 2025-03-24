import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import time
import wandb
import torch.nn.functional as F


def dropout(x, p=0.5, training=True):
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).type_as(x)
    return mask * x / (1.0 - p)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Currently using device:", device)


class ExponentialLearningRateDecay:
    def __init__(self, initial_lr, decay_rate=0.98):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def get_lr(self, epoch):
        return self.initial_lr * (self.decay_rate ** epoch)


class AdamOptimizer:
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.learning_rate = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.epsilon = epsilon
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if hasattr(p, 'grad') and p.grad is not None:
                # Adam accumulation
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (p.grad ** 2)

                # bias correction
                m_hat = self.m[i] / (1 - self.b1**self.t)
                v_hat = self.v[i] / (1 - self.b2**self.t)

                # param update
                p -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)


class FullyConnectedNN:
    def __init__(self, input_size, hidden_size, layers=1, learning_rate=0.001,
                 apply_activation_on_last=False, dropout=0.3):
        #self.learning_rate = learning_rate
        self.layers = layers
        self.apply_act_last = apply_activation_on_last
        self.dropout = dropout
        self.weights = []
        self.biases = []
        self.z = []
        self.a = []

        # initialize weights
        for _ in range(layers):
            std = math.sqrt(2.0 / (input_size + hidden_size))
            w = torch.randn(hidden_size, input_size) * std
            b = torch.zeros(hidden_size, 1)
            self.weights.append(w)
            self.biases.append(b)
            self.z.append(None)
            self.a.append(None)
            input_size = hidden_size

        self.x = None

    def activate(self, x):
        # ReLU
        return torch.clamp(x, min=0.0)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        self.x = x

        self.z = [None]*self.layers
        self.a = [None]*self.layers

        out = x
        for i in range(self.layers):
            z = out @ self.weights[i].T + self.biases[i].T
            if i < self.layers - 1 or self.apply_act_last:
                a = self.activate(z)
            else:
                a = z

            # dropout on all but final layer
            if i < self.layers - 1 and self.dropout > 0:
                a = dropout(a, p=self.dropout, training=True)

            self.z[i] = z
            self.a[i] = a
            out = a
        return out

    def backward(self, grad_out):
        for i in reversed(range(self.layers)):
            # derivative of ReLU
            if i < self.layers - 1 or self.apply_act_last:
                grad_out = grad_out * (self.z[i] > 0).float()

            prev_a = self.x if i == 0 else self.a[i - 1]
            dw = grad_out.T @ prev_a
            db = grad_out.sum(dim=0, keepdim=True).T

            dw = torch.clamp(dw, -1.0, 1.0)
            db = torch.clamp(db, -1.0, 1.0)

            self.weights[i].grad = dw
            self.biases[i].grad = db

            grad_out = grad_out @ self.weights[i]
        return grad_out

    def to(self, device):
        for i in range(self.layers):
            self.weights[i] = self.weights[i].to(device)
            self.biases[i] = self.biases[i].to(device)
        return self


class MaxPool:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x_shape = None
        self.indices = None

    def forward(self, x):
        self.x = x  # save for backprop
        self.x_shape = x.shape
        out, self.indices = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, return_indices=True)
        return out

    def backward(self, grad_output):
        return F.max_unpool2d(
            grad_output,
            self.indices,
            kernel_size=self.kernel_size,
            stride=self.stride,
            output_size=self.x_shape
        )

    def to(self, device):
        return self


class ConvolutionNN:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 learning_rate=0.001, dropout=0.3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / (fan_in + fan_out))

        self.W = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std
        self.b = torch.zeros(out_channels, 1)


        self.x = None
        self.x_unfold = None
        self.input_shape = None

    def forward(self, x):
        self.x = x
        self.input_shape = x.shape

        B, C, H, W = x.shape
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.x_unfold = x_unfolded

        W2d = self.W.view(self.out_channels, -1)
        out = W2d @ x_unfolded + self.b
        h_out = (H + 2*self.padding - self.kernel_size)//self.stride + 1
        w_out = (W + 2*self.padding - self.kernel_size)//self.stride + 1
        out = out.view(B, self.out_channels, h_out, w_out)


        if self.dropout > 0:
            out = dropout(out, p=self.dropout, training=True)
        return out

    def backward(self, grad_output):
        B, Cout, h_out, w_out = grad_output.shape
        grad_out_flat = grad_output.view(B, Cout, -1)
        W2d = self.W.view(Cout, -1)
        dW_2d = torch.zeros_like(W2d, device=grad_output.device)
        db = torch.zeros_like(self.b, device=grad_output.device)

        for b_idx in range(B):
            dW_2d += grad_out_flat[b_idx] @ self.x_unfold[b_idx].T
            db += grad_out_flat[b_idx].sum(dim=1, keepdim=True)

        dW_2d = torch.clamp(dW_2d, -1.0, 1.0)
        db = torch.clamp(db, -1.0, 1.0)

        dx_unfold = torch.zeros_like(self.x_unfold, device=grad_output.device)
        W_T = W2d.T
        for b_idx in range(B):
            dx_unfold[b_idx] = W_T @ grad_out_flat[b_idx]

        in_shape = self.input_shape[2], self.input_shape[3]
        grad_input = F.fold(dx_unfold, output_size=in_shape,
                            kernel_size=self.kernel_size, padding=self.padding,
                            stride=self.stride)

        self.W.grad = dW_2d.view_as(self.W)
        self.b.grad = db
        return grad_input

    def to(self, device):
        self.W = self.W.to(device)
        self.b = self.b.to(device)
        return self


def build(learning_rate=0.001, dropout=0.3):
    convs = [
        ConvolutionNN(3, 128, 3, 1, 1, learning_rate, dropout),
        MaxPool(kernel_size=2, stride=2),
        ConvolutionNN(128, 256, 3, 1, 1, learning_rate, dropout),
        MaxPool(kernel_size=2, stride=2),
    ]

    fcs = [
        FullyConnectedNN(256 * 8 * 8, 512, layers=1, learning_rate=learning_rate,
                         apply_activation_on_last=True, dropout=dropout),
        FullyConnectedNN(512, 256, layers=1, learning_rate=learning_rate,
                         apply_activation_on_last=True, dropout=dropout),
        FullyConnectedNN(256, 10, layers=1, learning_rate=learning_rate,
                         apply_activation_on_last=False, dropout=0.0)
    ]
    return convs, fcs


# train function
def train(conv_layers, fc_layers, loader, device, optimizer, all_params, l2_lambda=1e-4):
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, lables in tqdm(loader, desc="Training", leave=False):
        imgs, lables = imgs.to(device), lables.to(device)
        b_size = imgs.size(0)

        # forward pass conv
        out = imgs
        for m in conv_layers:
            out = m.forward(out)
        conv_out_shape = out.shape  # Capture the final conv output shape

        out = out.view(b_size, -1)

        for fc in fc_layers:
            out = fc.forward(out)

        
        logits = out
        """p = torch.exp(logits)
        p = p / p.sum(dim=1, keepdim=True)
        one_hot = torch.zeros_like(p)
        one_hot[range(b_size), b_size] = 1.0
        loss = -(one_hot * p.log()).sum(dim=1).mean()
        running_loss += loss.item()
        dlogits = (p - one_hot) / b_size"""


        max_vals = logits.max(dim=1, keepdim=True)[0]
        stable_logits = logits - max_vals
        exp_val = torch.exp(stable_logits)
        p = exp_val / exp_val.sum(dim=1, keepdim=True)

        one_hot = torch.zeros_like(p)
        one_hot[range(b_size), lables] = 1.0

        log_p = torch.log(p + 1e-12)
        loss = -(one_hot * log_p).sum(dim=1).mean()

        # L2 reg
        l2_term = sum(torch.sum(prm**2) for prm in all_params)
        loss = loss + l2_lambda * l2_term
        running_loss += loss.item()

        # compute grad w.r.t. logits
        dlogits = (p - one_hot) / b_size

        # Backward pass:
        fc_gradient = dlogits
        for fc in reversed(fc_layers):
            fc_gradient = fc.backward(fc_gradient)
        grad_conv = fc_gradient.view(conv_out_shape)
        for c in reversed(conv_layers):
            grad_conv = c.backward(grad_conv)

        optimizer.step()

        for param in all_params:
            param.grad = None

        preds = logits.argmax(dim=1)
        correct += (preds == lables).sum().item()
        total += lables.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def evaluate(conv_layers, fc_layers, loader, device, l2_lambda=1e-4):
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, lables in loader:
            imgs, lables = imgs.to(device), lables.to(device)
            b_size = imgs.size(0)

            out = imgs
            for c in conv_layers:
                out = c.forward(out)
            out = out.view(b_size, -1) # flatten
            for fc in fc_layers:
                out = fc.forward(out)

            """logits = out
            p = torch.exp(logits)
            p = p / p.sum(dim=1, keepdim=True)
            one_hot = torch.zeros_like(p)
            one_hot[range(b_size), lables] = 1.0
            loss = -(one_hot * p.log()).sum(dim=1).mean()
            test_loss += loss.item()"""


            logits = out
            stable_part = logits - logits.max(dim=1, keepdim=True)[0]
            e_lg = torch.exp(stable_part)
            p = e_lg / e_lg.sum(dim=1, keepdim=True)

            one_hot = torch.zeros_like(p)
            one_hot[range(b_size), lables] = 1.0

            log_p = torch.log(p + 1e-12)
            loss_val = -(one_hot * log_p).sum(dim=1).mean()

            # L2 reg
            tot_l2 = 0.0
            for m in conv_layers:
                if hasattr(m, 'W'):
                    tot_l2 += torch.sum(m.W**2)
                if hasattr(m, 'b'):
                    tot_l2 += torch.sum(m.b**2)
            for fc in fc_layers:
                if hasattr(fc, 'weights'):
                    for w in fc.weights:
                        tot_l2 += torch.sum(w**2)
                if hasattr(fc, 'biases'):
                    for b in fc.biases:
                        tot_l2 += torch.sum(b**2)

            loss_val = loss_val + l2_lambda * tot_l2
            test_loss += loss_val.item()

            preds = logits.argmax(dim=1)
            correct += (preds == lables).sum().item()
            total += lables.size(0)

    avg_loss = test_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def get_data_loaders1(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def get_data_loaders(batch_size=64):
    transf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])
    transf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, transform=transf_train, download=True)
    test_data = datasets.CIFAR10(root='./data', train=False, transform=transf_test, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, test_loader

# Main script
if __name__ == '__main__':
    num_epochs = 150
    base_lr = 0.001
    opt_type = 'adam'
    #architecture = ""
    drop_val = 0.0
    results = {}
    successful_exps = {}
    batch = 2048

    run_name = f"final"
    wandb.init(project="CSI4140DNN", name=run_name, reinit=True, config={
        #"architecture": architecture,
        "activation": "reLu",
        "dropout": drop_val,
        "learning_rate": base_lr,
        "optimizer": opt_type,
        "num_epochs": num_epochs
    })


    train_loader, test_loader = get_data_loaders(batch)
    conv_mods, fc_mods = build( learning_rate=base_lr, dropout=drop_val)

    # move to device
    for c in conv_mods:
        c.to(device)
    for f in fc_mods:
        f.to(device)

    # parameters
    all_params = []
    for c in conv_mods:
        if hasattr(c, 'W') and hasattr(c, 'b'):
            all_params.extend([c.W, c.b])
    for f in fc_mods:
        if hasattr(f, 'weights') and hasattr(f, 'biases'):
            for i in range(f.layers):
                all_params.append(f.weights[i])
                all_params.append(f.biases[i])

    # Adam optimizer
    optimizer = AdamOptimizer(all_params, learning_rate=base_lr)

    lr_sched = ExponentialLearningRateDecay(initial_lr=base_lr, decay_rate=0.98)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        start_epoch = time.time()
        # get new learning rate
        optimizer.learning_rate = lr_sched.get_lr(epoch)

        # train
        train_loss, train_acc = train(conv_mods, fc_mods, train_loader, device, optimizer, all_params)

        # test
        test_loss, test_acc = evaluate(conv_mods, fc_mods, test_loader, device)

        duration = time.time() - start_epoch
        total_elapsed = time.time() - start_time

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "epoch_time_sec": duration,
            "total_time_sec": total_elapsed
        })

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
            f"Ep Time: {duration:.2f}s, Total: {total_elapsed:.2f}s"
        )

    """results[(architecture, "reLu", drop_val)] = {
        "train_losses": tr_losses,
        "train_accs": tr_accs,
        "test_losses": tst_losses,
        "test_accs": tst_accs,
        "batch_size": batch
    }"""

    wandb.finish()
    """plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

    plt.figure()
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.show()"""
