import os
import numpy as np
from PIL import Image
import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from skimage.transform import resize

class evaluation_metrics:
    def __init__(self,img_path):
        self.img_path = img_path

    def visualize_images(self, similar_images_paths, query_image_path):
        query_image = Image.open(query_image_path)

        fig, axes = plt.subplots(1, len(similar_images_paths) + 1, figsize=(15, 5))
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

        for i, (file_name, distance) in enumerate(similar_images_paths, start=1):
            image = Image.open(os.path.join(self.img_path, str(file_name)+'.jpg'))
            axes[i].imshow(image)
            axes[i].set_title(f'Similar Image {i} ({file_name})')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show();


    def eval_results(self,query_image_path, similar_images_paths):
        query_image = np.array(Image.open(query_image_path).convert('RGB'))
        query_image = resize(query_image, (200, 200, 3))

        ssim_scores = []
        rmse_scores = []
        psnr_scores = []
        uqi_scores = []

        for (file_name,_) in similar_images_paths:
            img_path = os.path.join(self.img_path, str(file_name)+'.jpg')
            similar_image = np.array(Image.open(img_path).convert('RGB'))
            similar_image = resize(similar_image, (200, 200, 3))

            ssim_score, _ = ssim(query_image, similar_image, channel_axis=2, full=True,win_size=7,data_range=query_image.max() - query_image.min())
            ssim_scores.append(ssim_score)

            rmse_score = np.sqrt(mse(query_image, similar_image))
            rmse_scores.append(rmse_score)

        avg_ssim = np.mean(ssim_scores)
        avg_rmse = np.mean(rmse_scores)

        return avg_ssim, avg_rmse
