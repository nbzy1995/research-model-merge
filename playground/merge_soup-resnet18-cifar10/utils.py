import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


def create_cifar10_resnet18(num_classes=10, seed=42):
    """
    Create a ResNet18 model modified for CIFAR-10.

    Modifications:
    - Replace first conv layer (3x3 stride 1 instead of 7x7 stride 2)
    - Remove maxpool layer
    - Replace final FC layer for CIFAR-10 (10 classes)

    Args:
        num_classes: Number of output classes (default: 10)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Modified ResNet18 model
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    criterion,
    epochs,
    device,
    checkpoint_dir,
    checkpoint_name_template,
    log_interval=20,
    save_epoch_0=False
):
    """
    Train a model with validation and checkpoint saving.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler (or None)
        criterion: Loss function
        epochs: Number of epochs to train
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        checkpoint_name_template: Template for checkpoint names (must include {epoch})
        log_interval: How often to log batch progress
        save_epoch_0: Whether to save initial model before training

    Returns:
        Dictionary containing training history
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    # Save epoch 0 checkpoint if requested
    # TODO: we should evaluate the epoch 0 as well, and then save the checkpoints if requested. 
    if save_epoch_0:
        epoch_0_path = os.path.join(checkpoint_dir, checkpoint_name_template.format(epoch=0))
        save_checkpoint(model, epoch_0_path)
        print(f"✅ Saved epoch 0 checkpoint: {epoch_0_path}")

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # Training phase
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            train_batches += 1

            if i % log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })

        train_loss = train_loss_accum / train_batches
        history['train_loss'].append(train_loss)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {100*val_acc:.2f}%")
        print(f"  LR:         {current_lr:.6f}\n")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name_template.format(epoch=epoch+1))
        save_checkpoint(model, checkpoint_path)
        print(f"✅ Saved checkpoint: {checkpoint_path}\n")

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

    return history


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    loss_accum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='[Eval]', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss_accum += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = loss_accum / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def create_uniform_soup(state_dicts, weights=None):
    """
    Create a model soup by averaging state dicts.

    Args:
        state_dicts: List of state dictionaries to average
        weights: Optional list of weights for weighted average (default: uniform)

    Returns:
        Averaged state dictionary
    """
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    soup_state_dict = {k: v.clone() * weights[0] for k, v in state_dicts[0].items()}

    for i, state_dict in enumerate(state_dicts[1:], 1):
        for k, v in state_dict.items():
            soup_state_dict[k] += v.clone() * weights[i]

    return soup_state_dict


def compute_l2_distance(state_dict1, state_dict2):
    """
    Compute L2 distance between two model state dictionaries.

    Args:
        state_dict1: First state dictionary
        state_dict2: Second state dictionary

    Returns:
        L2 distance as a float
    """
    params1 = torch.cat([p.flatten() for p in state_dict1.values()])
    params2 = torch.cat([p.flatten() for p in state_dict2.values()])

    l2_dist = torch.norm(params1 - params2).item()

    return l2_dist


def save_checkpoint(model, path):
    """
    Save model state dict to file.

    Args:
        model: PyTorch model
        path: Path to save checkpoint
    """
    torch.save(model.state_dict(), path)


def load_checkpoint(path, device='cpu'):
    """
    Load model state dict from file.

    Args:
        path: Path to checkpoint file
        device: Device to load tensors to

    Returns:
        State dictionary
    """
    state_dict = torch.load(path, map_location=device)
    return state_dict
