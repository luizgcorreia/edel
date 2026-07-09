import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# Say we have 10 samples, 8 of which are all zeros, and 2 are non-zero
X = np.zeros((10, 5))
X[0] = [1.0, 2.0, 3.0, 4.0, 5.0]
X[1] = [5.0, 4.0, 3.0, 2.0, 1.0]

print("Original X:")
print(X)

# Center
mean = np.mean(X, axis=0)
X_centered = X - mean
print("\nMean vector:")
print(mean)

print("\nCentered X:")
print(X_centered)

# Normalize
X_norm = normalize(X_centered, axis=1, norm="l2")
print("\nNormalized X:")
print(X_norm)

# Perform PCA projection
pca = PCA(n_components=2)
coords = pca.fit_transform(X_norm)
print("\nProjected Coordinates (PCA):")
for i, coord in enumerate(coords):
    print(f"Row {i}: {coord}")
