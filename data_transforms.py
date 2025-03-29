from torchvision import transforms
from config import NORMALIZATION_MEAN, NORMALIZATION_STD, IMAGE_SIZE

def get_transforms(phase: str):
    if phase == "train":
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),  # Resize to target size
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
            transforms.RandomRotation(15),  # Randomly rotate images within ±15 degrees
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Random brightness, contrast, saturation, and hue adjustments
            transforms.RandomCrop(IMAGE_SIZE, padding=10),  # Random cropping with padding
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian blur
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),  # Normalize image
        ])
    elif phase == "val" or phase == "test":
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),  # Resize to target size
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),  # Normalize image
        ])
        