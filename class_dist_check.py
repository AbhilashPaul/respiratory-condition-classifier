from collections import Counter
import matplotlib.pyplot as plt

def check_class_distribution(train_loader, val_loader, test_loader, class_names):
    """
    Checks and visualizes the class distribution in training, validation, and test datasets.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        class_names (list): List of class names.

    Returns:
        None: Prints class distributions and visualizes them.
    """
    def get_label_counts(loader):
        """Helper function to count labels in a DataLoader."""
        label_counts = Counter()
        for _, labels in loader:
            label_counts.update(labels.tolist())
        return label_counts

    # Get label counts for training, validation, and test sets
    train_counts = get_label_counts(train_loader)
    val_counts = get_label_counts(val_loader)
    test_counts = get_label_counts(test_loader)

    # Print distributions
    print("Training Set Class Distribution:")
    for label, count in train_counts.items():
        print(f"{class_names[label]}: {count}")
    
    print("\nValidation Set Class Distribution:")
    for label, count in val_counts.items():
        print(f"{class_names[label]}: {count}")
    
    print("\nTest Set Class Distribution:")
    for label, count in test_counts.items():
        print(f"{class_names[label]}: {count}")

    # Visualize distributions
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].bar(train_counts.keys(), train_counts.values(), color='green', tick_label=[class_names[key] for key in train_counts.keys()])
    ax[0].set_title("Training Set Class Distribution")
    ax[0].set_xlabel("Class")
    ax[0].set_ylabel("Count")

    ax[1].bar(val_counts.keys(), val_counts.values(), color='orange', tick_label=[class_names[key] for key in val_counts.keys()])
    ax[1].set_title("Validation Set Class Distribution")
    ax[1].set_xlabel("Class")
    ax[1].set_ylabel("Count")

    ax[2].bar(test_counts.keys(), test_counts.values(), color='blue', tick_label=[class_names[key] for key in test_counts.keys()])
    ax[2].set_title("Test Set Class Distribution")
    ax[2].set_xlabel("Class")
    ax[2].set_ylabel("Count")

    plt.tight_layout(pad=3.0)

    plt.savefig('class_distribution.png', bbox_inches='tight')
    plt.show()
