import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, task='all', 
                num_artists=None, checkpoint_dir='/content/drive/MyDrive/checkpoints', start_epoch=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_acc = 0.0
    best_model_wts = model.state_dict().copy()

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = {'artist': 0, 'genre': 0, 'style': 0}
            total_samples = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                if inputs is None or labels is None or torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, task=task)
                    loss = 0

                    if task in ['all', 'artist']:
                        artist_loss = criterion(outputs['artist'], labels)
                        loss += artist_loss
                        _, artist_preds = torch.max(outputs['artist'], 1)
                        running_corrects['artist'] += torch.sum(artist_preds == labels.data)

                    if task in ['all', 'genre']:
                        genre_loss = criterion(outputs['genre'], labels)
                        loss += genre_loss
                        _, genre_preds = torch.max(outputs['genre'], 1)
                        running_corrects['genre'] += torch.sum(genre_preds == labels.data)

                    if task in ['all', 'style']:
                        style_loss = criterion(outputs['style'], labels)
                        loss += style_loss
                        _, style_preds = torch.max(outputs['style'], 1)
                        running_corrects['style'] += torch.sum(style_preds == labels.data)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / total_samples
            print(f'{phase} Loss: {epoch_loss:.4f}')

            task_acc = 0.0
            if task == 'artist':
                task_acc = running_corrects['artist'].double() / total_samples
                print(f'{phase} {task} Acc: {task_acc:.4f}')
            elif task == 'genre':
                task_acc = running_corrects['genre'].double() / total_samples
                print(f'{phase} {task} Acc: {task_acc:.4f}')
            elif task == 'style':
                task_acc = running_corrects['style'].double() / total_samples
                print(f'{phase} {task} Acc: {task_acc:.4f}')

            if phase == 'val' and task_acc > best_acc:
                best_acc = task_acc
                best_model_wts = model.state_dict().copy()
                checkpoint_path = os.path.join(checkpoint_dir, f'{task}_checkpoint_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")

        print()

    model.load_state_dict(best_model_wts)
    return model

def setup_training(model, num_epochs=15, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    return criterion, optimizer, scheduler