import os
import torch
from torch.utils.data import DataLoader
from src.models import ArtworkCRNN
from src.dataset import ArtworkDataset, get_transforms
from src.train import train_model, setup_training
from src.utils import find_outliers, visualize_tsne, save_outliers_to_file
from src.evaluation import evaluate_model

def run_artwork_classification(
    extract_path,
    zip_file_name="wikiart.zip",
    batch_size=32,
    num_epochs=15,
    learning_rate=0.001,
    resume_from_checkpoint=None
):
    zip_file_path = os.path.join(extract_path, zip_file_name)
    artist_class_file = os.path.join(extract_path, "artist_class.txt")
    genre_class_file = os.path.join(extract_path, "genre_class.txt")
    style_class_file = os.path.join(extract_path, "style_class.txt")

    artist_train_csv = os.path.join(extract_path, "artist_train.csv")
    artist_val_csv = os.path.join(extract_path, "artist_val.csv")

    genre_train_csv = os.path.join(extract_path, "genre_train.csv")
    genre_val_csv = os.path.join(extract_path, "genre_val.csv")

    style_train_csv = os.path.join(extract_path, "style_train.csv")
    style_val_csv = os.path.join(extract_path, "style_val.csv")

    data_transforms = get_transforms()

    artist_train_dataset = ArtworkDataset(artist_train_csv, None, artist_class_file, data_transforms['train'], extract_path=extract_path)
    artist_val_dataset = ArtworkDataset(artist_val_csv, None, artist_class_file, data_transforms['val'], extract_path=extract_path)

    genre_train_dataset = ArtworkDataset(genre_train_csv, None, genre_class_file, data_transforms['train'], extract_path=extract_path)
    genre_val_dataset = ArtworkDataset(genre_val_csv, None, genre_class_file, data_transforms['val'], extract_path=extract_path)

    style_train_dataset = ArtworkDataset(style_train_csv, None, style_class_file, data_transforms['train'], extract_path=extract_path)
    style_val_dataset = ArtworkDataset(style_val_csv, None, style_class_file, data_transforms['val'], extract_path=extract_path)

    artist_train_loader = DataLoader(artist_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    artist_val_loader = DataLoader(artist_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    genre_train_loader = DataLoader(genre_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    genre_val_loader = DataLoader(genre_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    style_train_loader = DataLoader(style_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    style_val_loader = DataLoader(style_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_artists = artist_train_dataset.num_classes
    num_genres = genre_train_dataset.num_classes
    num_styles = style_train_dataset.num_classes

    print(f"Number of artist classes: {num_artists}")
    print(f"Number of genre classes: {num_genres}")
    print(f"Number of style classes: {num_styles}")

    model = ArtworkCRNN(num_artists, num_genres, num_styles, pretrained=True)

    tasks = ['artist', 'genre', 'style']
    
    for task in tasks:
        print(f"\nTraining for task: {task}")
        
        if task == 'artist':
            train_loader = artist_train_loader
            val_loader = artist_val_loader
        elif task == 'genre':
            train_loader = genre_train_loader
            val_loader = genre_val_loader
        else:  
            train_loader = style_train_loader
            val_loader = style_val_loader

        criterion, optimizer, scheduler = setup_training(model, num_epochs, learning_rate)
        
        model = train_model(
            model, 
            {'train': train_loader, 'val': val_loader}, 
            criterion, 
            optimizer, 
            scheduler, 
            num_epochs=num_epochs, 
            task=task, 
            num_artists=num_artists
        )

        torch.save(model.state_dict(), f'{task}_model.pth')

    outliers = []
    
    artist_outliers = find_outliers(model, artist_val_loader, artist_val_dataset)
    genre_outliers = find_outliers(model, genre_val_loader, genre_val_dataset)
    style_outliers = find_outliers(model, style_val_loader, style_val_dataset)
    
    outliers.extend(artist_outliers)
    outliers.extend(genre_outliers)
    outliers.extend(style_outliers)

    save_outliers_to_file(outliers, "outlier_images.txt")

    class_mappings = {
        'artist': {i: artist_train_dataset.get_class_name(i) for i in range(num_artists)},
        'genre': {i: genre_train_dataset.get_class_name(i) for i in range(num_genres)},
        'style': {i: style_train_dataset.get_class_name(i) for i in range(num_styles)}
    }

    return model, outliers, class_mappings

if __name__ == "__main__":
    model, outliers, class_mappings = run_artwork_classification("/path/to/dataset")