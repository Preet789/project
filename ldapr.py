import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the extracted features
extracted_features = np.load('extracted_features.npy')

# Load the labels
labels = [stored_data[i][b'labels'] for i in range(len(stored_data))]

# Initialize the LDA model
lda = LinearDiscriminantAnalysis()

# Fit the LDA model to the data
lda.fit(extracted_features, labels)

# Compute explained variance ratio for each component
explained_variance_ratio = lda.explained_variance_ratio_

# Compute cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Plot the cumulative explained variance
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Number of Components')
plt.grid(True)
plt.show()

# Choose the smallest n_components that captures a satisfactory amount of variance (e.g., 99%)
desired_variance = 0.99
n_components_satisfactory = np.argmax(cumulative_explained_variance >= desired_variance) + 1
print("Number of components capturing {}% of variance: {}".format(desired_variance * 100, n_components_satisfactory))
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the extracted features
extracted_features = np.load('extracted_features.npy')

# Load the labels
labels = [stored_data[i][b'labels'] for i in range(len(stored_data))]

# Initialize the LDA model
lda = LinearDiscriminantAnalysis(n_components=9)  # You can adjust the number of components as needed

# Fit the LDA model to the data
lda.fit(extracted_features, labels)

# Transform the features to the lower-dimensional space
features_lda = lda.transform(extracted_features)

# Now 'features_lda' contains the features after LDA dimensionality reduction
print(features_lda.shape)  # Output: (1000, 9)

# You can now use 'features_lda' for further analysis, such as visualization, clustering, or classification
