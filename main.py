import os
import torch
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
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


def load_and_split_dataset(data_dir, train_split=0.8, batch_size=8):
    """
    Loads a dataset from the given directory and splits it into training and test sets with separate transformations.

    Args:
        data_dir (str): Path to the root directory of the dataset.
        train_split (float): Proportion of data to use for training (default is 0.8).
        batch_size (int): Batch size for DataLoaders.
        transforms_dict (dict): Dictionary containing 'train_transform' and 'test_transform'.

    Returns:
        dict: A dictionary containing DataLoaders for training and testing datasets,
              along with class-to-index mapping.
    """
 
    # Load datasets with separate transformations
    train_dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms("train"))
    val_dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms("val"))

    # Calculate split sizes
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # Split the dataset into training and val sets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders for both sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Return a dictionary with DataLoaders and class-to-index mapping
    return train_loader, test_loader, train_dataset.dataset.class_to_idx, len(train_dataset), len(val_dataset)

if __name__ == "__main__":
    # Load dataset and split into train/val sets
    train_loader, val_loader, class_to_idx, train_size, val_size= load_and_split_dataset(data_dir=os.path.join(data_dir, 'train'), train_split=0.8, batch_size=BATCH_SIZE)
    # Print results
    print("Training Dataset Size:", train_size)
    print("Testing Dataset Size:", val_size)
    dataloaders = {"train":train_loader, "val":val_loader}
    data_sizes = {"train": train_size, "val": val_size}
    class_names = list(class_to_idx.keys())
    print("Class Names:", class_names)

    model = get_model(class_names)
    model = model.to(DEVICE)
    #print("Model Summary:")
    #print(model)

    # specify loss function (categorical cross-entropy loss)
    criterion = nn.CrossEntropyLoss() 

    # Specify optimizer which performs Gradient Descent
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Learning Scheduler

    trained_model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, data_sizes, num_epochs=NUM_EPOCHS)

    # Save the trained model
    torch.save(trained_model.state_dict(), saved_model_path)
    print(f"Model saved to {saved_model_path}")


    test_image = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=get_transforms('test'))
    test_dataloader = DataLoader(test_image, batch_size=1, shuffle=False)

    y_pred_list, y_true_list = evaluate_model(test_dataloader, class_names)
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

