import torch
# Data loading
saved_model_path = "output/best_model.pth"
data_dir = "xray_dataset"

IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
