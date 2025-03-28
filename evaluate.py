import torch
from config import DEVICE, saved_model_path
import tqdm
from model import get_model

def evaluate_model(test_dataloader, class_names):
    model = get_model(class_names)
    # Load the saved weights
    model.load_state_dict(torch.load(saved_model_path))

    model = model.to(DEVICE)

    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_dataloader, desc='test', leave=False):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels
