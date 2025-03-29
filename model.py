import torch.nn as nn
from torchvision import models

class CNNModel(nn.Module):
    def __init__(self, class_names):
        super(CNNModel, self).__init__()
        
        # Load pre-trained DenseNet121 model
        self.base_model = models.densenet121(weights='IMAGENET1K_V1')
        
        # Get the number of features output from CNN layer
        num_ftrs = self.base_model.classifier.in_features
        
        # Replace the classifier layer
        self.base_model.classifier = nn.Linear(num_ftrs, len(class_names))

    def forward(self, x):
        return self.base_model(x)
    
class LightEfficientNetV2(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the EfficientNetV2-S model for binary classification.

        Args:
            num_classes (int): Number of output classes (e.g., 2 for COVID detection).
        """
        super(LightEfficientNetV2, self).__init__()
        # Load the pre-trained EfficientNetV2-S model
        self.model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        
        # Replace the classifier layer to match the number of classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Add dropout for regularization
            nn.Linear(in_features, len(num_classes))  # Fully connected layer for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor (e.g., batch of images).

        Returns:
            torch.Tensor: Output predictions.
        """
        return self.model(x)

def get_model(class_names):
    return LightEfficientNetV2(class_names)
