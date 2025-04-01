import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloader, task='all', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs, task=task)
            
            # Get predictions based on task
            if task == 'artist':
                _, preds = torch.max(outputs['artist'], 1)
            elif task == 'genre':
                _, preds = torch.max(outputs['genre'], 1)
            elif task == 'style':
                _, preds = torch.max(outputs['style'], 1)
            else:  # 'all' task
                _, preds = torch.max(outputs[task], 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        'classification_report': classification_report(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }