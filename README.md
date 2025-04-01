# humanai_gsoc25_tasks

## Overview
This repository contains solutions to tasks related to the Google Summer of Code (GSoC) 2025 application process for ArtExtract and SIRA Projects at HumanAI Organization. The tasks involve application of machine learning and deep learning solutions on different forms of art.

## Repository Structure
<img width="310" alt="Screenshot 2025-04-01 at 10 17 23 PM" src="https://github.com/user-attachments/assets/3dec33fd-1738-436f-8a86-d6625e7e9bad" />

## Tasks and Implementation
### Task 1: Convolutional-Recurrent Architectures
#### Description
Build a model based on convolutional-recurrent architectures for classifying Style, Artist, Genre, and other attributes. The task requires selecting the most appropriate approach and discussing the strategy used.
#### Dataset
https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view
#### Solution Approach
* Used resnet18 as a feature extractor, replacing the final layers with classification heads.
* Implemented a hybrid CNN-RNN model to capture spatial and sequential dependencies.
* Applied transfer learning by fine-tuning pre-trained ResNet18 weights.
* Trained using the train_model() function, which includes optimizer scheduling, checkpoint saving, and periodic evaluation.
#### Evaluation Metrics
* accuracy_score, confusion_matrix, and classification_report to assess classification performance.
* Used t-SNE visualization to analyze embedding spaces.
* Outlier detection to identify paintings that deviate from the assigned artist or genre.
#### Results
<img width="440" alt="Screenshot 2025-04-01 at 10 43 38 PM" src="https://github.com/user-attachments/assets/f84c0138-e231-4be5-9322-7d24b2c36201" />

#### Improvements
* Can increase the training epochs with abundant computational resources.
* Implementation of optimized transformer-based model architecture (ViT).
* Improve data handling.

### Task 2: Similarity
#### Description
Build a model to find similarities in paintings, focusing on identifying portraits with similar faces or poses. Use the National Gallery of Art open dataset: https://github.com/NationalGalleryOfArt/opendata
#### Solution Approach
* Data Preprocessing:
  * Download images and metadata using the provided ImageDownloader class.
  * Preprocess images by resizing, normalizing, and converting them to tensor format.
  * Extract facial regions using a pre-trained face detection model (e.g., MTCNN or OpenCV).
* Feature Extraction:
  * Utilize a deep learning model such as ResNet50 or EfficientNet as a feature extractor.
  * Extract embeddings from the penultimate layer of the CNN.
  * Normalize embeddings for similarity computation.
* Similarity Computation:
  * Compute pairwise similarities using cosine similarity or Euclidean distance.
  * Store embeddings in a vector database (e.g., FAISS) for efficient retrieval.
* Model Training & Optimization:
  * Fine-tune a Siamese Network or a Triplet Network for metric learning.
  * Train using contrastive loss or triplet loss for improved representation learning.
#### Evaluation Metrics
* Cosine Similarity Score: Measures similarity between feature embeddings.
* Precision-Recall at K: Evaluates retrieval performance.
* Structural Similarity Index (SSIM) and Root Mean Square Error (RMSE)
#### Results
<img width="784" alt="Screenshot 2025-04-01 at 11 01 17 PM" src="https://github.com/user-attachments/assets/cd156eb7-caf7-40c7-9b84-2de474cbfaaf" />

#### Improvements
* Can increase the percent of data used for the implementation.
* Can implement facial feature extraction.

## Setup Instructions
* Clone the repository.
* Extract the dataset in a proper format for the corresponding task.
* Create a virtual environment.
* Install the required libraries.
* Run the main.py script.

## Discuss
Due to several constraints in terms of time and available computational resources, I was only able to implement a very basic version of both tasks. However, I have included my thought process for potential improvements that I would love to implement in the near future. Thanks!
