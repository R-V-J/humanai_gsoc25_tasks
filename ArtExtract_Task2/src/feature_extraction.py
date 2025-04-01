import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import transforms
import pickle

class ImageRetrieval:
    def __init__(self, compressor, img_path, k=5, face_crop=False, device='cpu'):
        self.compressor = compressor
        self.compressor.device = device
        self.merged_df = pd.read_csv('./data/merged.csv')
        self.img_path = img_path
        self.device = device
        self.k = k

        pickle_file = "features.pkl"
        if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0: # Check if the file exists and is not empty
            with open(pickle_file, "rb") as f:
                self.features, self.image_paths = pickle.load(f)
            print(f"Loaded features from {pickle_file}")
        else:
            print(f"features.pkl is empty or does not exist. Extracting features...")
            self.features, self.image_paths = self.extract_features(face_crop)

    def extract_features(self, face_crop=False, save_interval=100000):
      pickle_file = "features.pkl"
      if os.path.exists(pickle_file):
          with open(pickle_file, "rb") as f:
              features, image_paths = pickle.load(f)

          total_images = len(self.merged_df)

          if len(image_paths) == total_images:
              print(f"Loaded features from {pickle_file}. Feature extraction is already complete.")
              return features, image_paths
          else:
              print(f"Loaded features for {len(image_paths)} images from {pickle_file}. Resuming feature extraction.")
              start_index = len(image_paths)
      else:
          features = []
          image_paths = []
          start_index = 0

      image_dataset = ImageDataset(self.merged_df)
      data_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)

      if face_crop:
          face_cropper = FaceCropper(self.compressor, device=self.device)

      print('Building Feature Vector List')
      for i, image_tensors in enumerate(tqdm(data_loader, total=len(data_loader), initial=start_index // data_loader.batch_size)):
          if i < start_index // data_loader.batch_size:
              continue

          if face_crop:
              extracted_features = face_cropper.crop_faces(image_tensors)
          else:
              extracted_features = self.compressor.extract(image_tensors.to(self.device))

          extracted_features = extracted_features.view(extracted_features.size(0), -1).cpu().numpy()
          features.extend(extracted_features)
          image_paths.extend(image_dataset.dataFrame.iloc[len(features) - len(extracted_features):len(features)]['objectid'].tolist())

          if len(image_paths) % save_interval == 0:
              with open(pickle_file, "wb") as f:
                  pickle.dump((features, image_paths), f)
              print(f"Saved progress to {pickle_file}")

      return features, image_paths

    def retrieve_similar_images(self, query_image_path, metric='cosine',face_crop=False):
        self.transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((200,200)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        image = Image.open(query_image_path).convert("RGB")
        query_image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        if face_crop:
            face_cropper = FaceCropper(self.compressor,device=self.device)
            query_features = face_cropper.crop_faces(query_image_tensor)
            if query_features is None:
                    query_features = self.compressor.extract(query_image_tensor)
        else:
            query_features = self.compressor.extract(query_image_tensor)

        query_object_id = os.path.basename(query_image_path).split(".")[0]
        query_index = self.merged_df[self.merged_df['objectid'] == int(query_object_id)].index.tolist()[0]
        except_query = np.delete(self.features, query_index+1, axis=0)

        knn = NearestNeighbors(n_neighbors=self.k, metric=metric)
        knn.fit(except_query)

        query_features = query_features.cpu().numpy()
        distances, indices = knn.kneighbors(query_features)
        similar_images = [(self.image_paths[i], distances[0, j]) for j, i in enumerate(indices[0])]
        similar_images = similar_images[1:]  # Skip the first element (query image)
        similar_images.sort(key=lambda x: x[1])  # Sort the similar images by distance

        return similar_images