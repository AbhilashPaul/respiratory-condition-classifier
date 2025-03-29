import torch.nn as nn
from torchvision import models

class Densenet121Model(nn.Module):
    def __init__(self, class_names):
        super(Densenet121Model, self).__init__()
        self.model = models.densenet121(weights='IMAGENET1K_V1')
        in_features = self.model.classifier.in_features
        
        # Replace the classifier layer
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Add dropout for regularization
            nn.Linear(in_features, len(class_names))
        )

    def forward(self, x):
        return self.model(x)
    
class LightEfficientNetV2(nn.Module):
    def __init__(self, class_names):
        super(LightEfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        
        # Replace the classifier layer to match the number of classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Add dropout for regularization
            nn.Linear(in_features, len(class_names))
        )

    def forward(self, x):
        return self.model(x)

def get_model(class_names):
    return LightEfficientNetV2(class_names)
