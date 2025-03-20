import torch
import wandb  # <-- add this
from data_loader import get_cifar10_loaders
from layers import (
    Conv2DLayer,
    FullyConnectedLayer,
    ReLU,
    Dropout,
    SoftmaxCrossEntropy,
    ResidualBlock
)
from model import Model
from train import train_one_epoch, evaluate
from optimizers import SGDWithMomentum, Adam, CosineLRDecay, ExponentialLRDecay
from utils import plot_metrics
from regularization import l2_regularization, l2_grad

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # ---------------------
    # 1) INITIALIZE WANDB
    # ---------------------
    # Typically you give a 'project' name so all runs are grouped there.
    # Optionally, you can specify run name, config, etc.
    wandb.init(project="my_cifar10_project", name="run_001")  # or remove 'name' if you want a random name

    # If you want to log hyperparameters systematically:
    config = wandb.config
    config.batch_size = 128
    config.epochs = 20
    config.learning_rate = 0.01
    config.lambda_l2 = 1e-4
    config.momentum = 0.9

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config.batch_size,
        augment=True
    )

    # Example architecture
    layers = [
        Conv2DLayer(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, device=device),
        ReLU(),
        ResidualBlock(16, 16),
        Dropout(p=0.3),
        FullyConnectedLayer(16*32*32, 10, device=device)
    ]
    model = Model(layers)

    # Loss function
    criterion = SoftmaxCrossEntropy()

    # Optimizer
    optimizer = SGDWithMomentum(
        model.params_and_grads(),
        lr=config.learning_rate,
        momentum=config.momentum
    )

    # LR scheduler
    scheduler = CosineLRDecay(optimizer, config.learning_rate, config.epochs)

    train_accs, test_accs = [], []
    train_losses, test_losses = [], []

    for epoch in range(config.epochs):
        # Decay LR if using a scheduler
        scheduler.step(epoch)

        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            lambda_=config.lambda_l2
        )

        # Test
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion
        )

        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # ---------------------
        # 2) LOG METRICS TO WANDB
        # ---------------------
        # Log what you want: typically epoch number, train/test losses, accuracies, etc.
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "learning_rate": optimizer.lr  # if your optimizer stores current lr in .lr
        })

        # (Optional) print metrics
        print(f"Epoch [{epoch+1}/{config.epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}")

    # If you want to plot locally, you can still do that:
    # plot_metrics(train_accs, test_accs, train_losses, test_losses)

    # ---------------------
    # 3) FINISH WANDB RUN
    # ---------------------
    wandb.finish()

if __name__ == "__main__":
    main()
