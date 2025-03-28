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

def get_model(class_names):
    return CNNModel(class_names)
