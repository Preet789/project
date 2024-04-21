import numpy as np

import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os  # Import os module for file operations
import requests

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import pickle
features_lda = np.load('features_lda.npy')
labels = np.load('labels.npy')
with open('stored_data.pkl', 'rb') as f:
    stored_data = pickle.load(f)
with open('lda.pkl', 'rb') as f:
    lda = pickle.load(f)
with open('svm_classifier.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)
resnet = models.resnet50(pretrained=True)
# Remove the last fully connected layer
resnet = nn.Sequential(*list(resnet.children())[:-1])
# Set the model to evaluation mode
resnet.eval()

def extract_features_from_image(input_image, model, lda):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(input_image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    features = torch.flatten(features)
    features = lda.transform(features.unsqueeze(0))
    return features[0]

def save_similar_images(similar_image_indices, stored_data, output_dir):
    similar_image_paths = []
    for i, index in enumerate(similar_image_indices):
        image_data = np.stack((stored_data[index][b'R'], stored_data[index][b'G'], stored_data[index][b'B']), axis=-1)
        similar_image = Image.fromarray(image_data)
        # Define the path to save the image
        image_path = os.path.join(output_dir, f'similar_image_{i + 1}.jpg')
        # Save the image
        similar_image.save(image_path)
        similar_image_paths.append(image_path)
    return similar_image_paths

def retrieve_similar_images_from_input_image(input_image_features, k=5, output_dir='similar_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    predicted_label = svm_classifier.predict([input_image_features])[0]
    similar_indices = [i for i, label in enumerate(labels) if label == predicted_label]
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(features_lda[similar_indices])
    distances, indices = nn_model.kneighbors([input_image_features])
    original_indices = [similar_indices[i] for i in indices[0]]
    
    # Save similar images locally and return their paths
    similar_image_paths = save_similar_images(original_indices, stored_data, output_dir)
    return similar_image_paths

# def visualize_retrieved_images(input_image, similar_image_paths):
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 6, 1)
#     plt.imshow(input_image)
#     plt.title('Input Image')
#     plt.axis('off')

#     for i, image_path in enumerate(similar_image_paths):
#         plt.subplot(1, 6, i + 2)
#         image_data = np.array(Image.open(image_path))
#         plt.imshow(image_data)
#         plt.title(f'Similar Image {i + 1}')
#         plt.axis('off')

#     plt.show()

def image_retrieval(input_image_path, resnet, lda):
    input_image = Image.open(input_image_path)
    input_image_features = extract_features_from_image(input_image, resnet, lda)
    similar_image_paths = retrieve_similar_images_from_input_image(input_image_features)
    # visualize_retrieved_images(input_image, similar_image_paths)
    return(similar_image_paths)
input_image_path = r'C:\Users\Preet\Desktop\ml\project\download (8).jpeg'
similar_image_paths=image_retrieval(input_image_path, resnet, lda)
print(similar_image_paths)