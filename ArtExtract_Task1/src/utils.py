import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def find_outliers(model, dataloader, dataset, threshold=0.7):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    outliers = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)

            outputs = model(inputs)

            for task in ['artist', 'genre', 'style']:
                if outputs[task] is not None:
                    task_labels = labels.to(device)
                    probabilities = torch.softmax(outputs[task], dim=1)
                    max_probs, preds = torch.max(probabilities, dim=1)

                    correct_mask = (preds == task_labels)
                    low_conf_mask = (max_probs < threshold)

                    for i in range(len(inputs)):
                        if low_conf_mask[i]:
                            img_path = dataset.data_frame.iloc[batch_idx * dataloader.batch_size + i, 0]
                            outliers.append({
                                'image_idx': i,
                                'task': task,
                                'true_label': task_labels[i].item(),
                                'predicted': preds[i].item(),
                                'confidence': max_probs[i].item(),
                                'features': outputs['features'][i].cpu().numpy(),
                                'image_path': img_path
                            })

    return outliers

def visualize_tsne(features, labels, class_mapping, title):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    unique_classes = np.unique(labels)

    for class_idx in unique_classes:
        mask = labels == class_idx
        class_name = class_mapping.get(class_idx, f"Class {class_idx}")
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=class_name, alpha=0.6)

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def save_outliers_to_file(outliers, file_path):
    with open(file_path, "w") as f:
        for outlier in outliers:
            f.write(outlier['image_path'] + "\n")