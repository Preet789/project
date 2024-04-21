# app.py

from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Importing necessary libraries and functions for image retrieval
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import pickle

def get_similar_images(image_path, resnet, lda):
    # Load pre-trained models and data
    features_lda = np.load('features_lda.npy')
    labels = np.load('labels.npy')
    with open('stored_data.pkl', 'rb') as f:
        stored_data = pickle.load(f)
    with open('lda.pkl', 'rb') as f:
        lda = pickle.load(f)
    with open('svm_classifier.pkl', 'rb') as f:
        svm_classifier = pickle.load(f)
    resnet = models.resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    # Function to extract features from an input image
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

    # Function to save similar images
    def save_similar_images(similar_image_indices, stored_data, output_dir):
        similar_image_paths = []
        for i, index in enumerate(similar_image_indices):
            subdirectory = os.path.join(output_dir, f'similar_images_{i+1}')
            if not os.path.exists(subdirectory):
                os.makedirs(subdirectory)
            
            image_data = np.stack((stored_data[index][b'R'], stored_data[index][b'G'], stored_data[index][b'B']), axis=-1)
            similar_image = Image.fromarray(image_data)
            image_path = os.path.join(subdirectory, f'similar_image_{i + 1}.jpg')
            similar_image.save(image_path)
            similar_image_paths.append(image_path)
        return similar_image_paths

    # Function to retrieve similar images based on the input image
    def retrieve_similar_images_from_input_image(input_image_features, k=5, output_dir='static/similar_images'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        predicted_label = svm_classifier.predict([input_image_features])[0]
        similar_indices = [i for i, label in enumerate(labels) if label == predicted_label]
        nn_model = NearestNeighbors(n_neighbors=k)
        nn_model.fit(features_lda[similar_indices])
        _, indices = nn_model.kneighbors([input_image_features])
        original_indices = [similar_indices[i] for i in indices[0]]
        
        # Save similar images locally and return their paths
        similar_image_paths = save_similar_images(original_indices, stored_data, output_dir)
        return similar_image_paths

    # Function to retrieve similar images and return their paths
    def image_retrieval(input_image_path, resnet, lda):
        input_image = Image.open(input_image_path)
        input_image_features = extract_features_from_image(input_image, resnet, lda)
        similar_image_paths = retrieve_similar_images_from_input_image(input_image_features)
        return similar_image_paths

    # Call the image retrieval function and return the similar image paths
    similar_image_paths = image_retrieval(image_path, resnet, lda)

    return similar_image_paths

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('show_results', filename=file.filename))
    return render_template('upload.html')

@app.route('/results/<filename>')
def show_results(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Load resnet and lda models
    resnet = models.resnet50(pretrained=True)
    lda = pickle.load(open('lda.pkl', 'rb'))
    
    # Pass resnet and lda to get_similar_images
    similar_images = get_similar_images(path, resnet, lda)
    return render_template('results.html', similar_images_paths=similar_images)



if __name__ == '__main__':
    app.run(debug=True)
