import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from data_transforms import get_transforms
from config import BATCH_SIZE, data_dir
from model import get_model
import torch.nn as nn
import torch.optim as optim
from train import train_model
from config import NUM_EPOCHS, saved_model_path, DEVICE
from evaluate import evaluate_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from class_dist_check import check_class_distribution

def load_and_split_dataset(data_dir, train_split=0.7, val_split=0.2, batch_size=8):
    """
    Loads a dataset from the given directory and splits it into training, validation, and test sets with separate transformations.

    Args:
        data_dir (str): Path to the root directory of the dataset.
        train_split (float): Proportion of data to use for training (default is 0.7).
        val_split (float): Proportion of data to use for validation (default is 0.2).
        batch_size (int): Batch size for DataLoaders.

    Returns:
        dict: DataLoaders for training, validation, and testing datasets, class-to-index mapping, and dataset sizes.
    """
    # Load the full dataset with training transformations initially
    full_dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms("train"))

    # Extract labels for stratified splitting
    labels = [sample[1] for sample in full_dataset.samples]  # Get class labels

    # Calculate split sizes
    train_size = int(train_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Initialize StratifiedShuffleSplit for train/val split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=42)

    for train_idx, temp_idx in stratified_split.split(full_dataset.samples, labels):
        train_dataset = Subset(full_dataset, train_idx)
        temp_dataset = Subset(full_dataset, temp_idx)

        # Extract labels for validation/test split
        temp_labels = [labels[i] for i in temp_idx]

        # Initialize StratifiedShuffleSplit for validation/test split
        stratified_temp_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

        for val_idx, test_idx in stratified_temp_split.split(temp_labels, temp_labels):
            val_dataset = Subset(temp_dataset, val_idx)
            test_dataset = Subset(temp_dataset, test_idx)

    # Apply separate transforms to validation and test datasets
    val_dataset.dataset.transform = get_transforms("val")
    test_dataset.dataset.transform = get_transforms("test")

    # Create DataLoaders for all sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return DataLoaders and metadata
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_to_idx": full_dataset.class_to_idx,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }

if __name__ == "__main__":
    # Load dataset and split into train/val sets
    dataset_info = load_and_split_dataset(data_dir=os.path.join(data_dir, 'train'), train_split=0.7, val_split=0.2, batch_size=BATCH_SIZE)
    train_loader = dataset_info["train_loader"]
    val_loader = dataset_info["val_loader"]
    test_loader = dataset_info["test_loader"]
    class_to_idx = dataset_info["class_to_idx"]
    train_size = dataset_info["train_size"]
    val_size = dataset_info["val_size"]
    test_size = dataset_info["test_size"]

    # Print results
    print("Training Dataset Size:", train_size)
    print("Validation Dataset Size:", val_size)
    print("Testing Dataset Size:", test_size)
    dataloaders = {"train":train_loader, "val":val_loader}
    data_sizes = {"train": train_size, "val": val_size}
    class_names = list(class_to_idx.keys())
    print("Class Names:", class_names)

    check_class_distribution(train_loader, val_loader, test_loader, class_names)

    model = get_model(class_names)
    model = model.to(DEVICE)
    #print("Model Summary:")
    #print(model)

    # specify loss function (categorical cross-entropy loss)
    criterion = nn.CrossEntropyLoss() 

    # Specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Use ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Monitor validation loss (minimization)
        factor=0.1,           # Reduce LR by multiplying by this factor
        patience=2,           # Wait for 2 epochs with no improvement
        threshold=0.0001,     # Minimum change in monitored quantity to qualify as improvement
        threshold_mode='rel', # Relative change (default)
        cooldown=0,           # Cooldown period before resuming normal operation
        min_lr=1e-6,          # Minimum learning rate
    )

    trained_model = train_model(model, criterion, optimizer, scheduler, dataloaders, data_sizes, num_epochs=NUM_EPOCHS)

    # Save the trained model
    torch.save(trained_model.state_dict(), saved_model_path)
    print(f"Model saved to {saved_model_path}")

    # Evaluate model
    y_pred_list, y_true_list = evaluate_model(test_loader, class_names)
    print("Unique classes in y_true_list:", np.unique(y_true_list))
    print("Unique classes in y_pred_list:", np.unique(y_pred_list))

    # print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_true_list, y_pred_list):.4f}")
    print(classification_report(y_true_list, y_pred_list, target_names=class_names))

    # plot confusion matrix
    cmatrix_test_set = confusion_matrix(y_true_list, y_pred_list)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cmatrix_test_set,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    plt.savefig('confusion_martrix.png', bbox_inches='tight')
    plt.show()

