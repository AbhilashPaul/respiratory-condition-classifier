import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 20
NUM_EPOCHS = 30

saved_model_path = "output/best_model.pth"
data_dir = "xray_dataset"
