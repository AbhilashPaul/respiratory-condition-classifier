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

class VGG16Model(nn.Module):
    def __init__(self, class_names):
        super(VGG16Model, self).__init__()
        
        # Load pre-trained VGG16 with updated weights parameter
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Freeze feature extraction layers
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Replace classifier while preserving original structure
        in_features = self.model.classifier[6].in_features  # 4096 features from last FC layer
        self.model.classifier[6] = nn.Sequential(
            nn.Dropout(p=0.5),  # Keep original dropout rate
            nn.Linear(in_features, len(class_names))
        )

    def forward(self, x):
        return self.model(x)
    

def get_model(class_names):
    return VGG16Model(class_names)
