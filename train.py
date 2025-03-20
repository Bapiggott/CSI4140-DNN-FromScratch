import torch

def train_one_epoch(model, train_loader, criterion, optimizer, lambda_=0.0):
    """
    - For each batch in train_loader:
        1) forward pass
        2) compute loss + add L2 if needed
        3) backward pass
        4) add L2 grad if needed
        5) optimizer step
    - Track average loss, accuracy, etc.
    Returns: (avg_epoch_loss, avg_epoch_accuracy)
    """

    """maybe try:
    for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    """
    pass

def evaluate(model, data_loader, criterion):
    """
    Evaluate model on a dataset (test/valid set).
    Return (avg_loss, avg_accuracy).
    """
    pass
