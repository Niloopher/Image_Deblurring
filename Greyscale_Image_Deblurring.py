import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


from scipy.linalg import circulant, kron
from scipy.ndimage import gaussian_filter1d

# Load and convert image
img = cv2.imread('na.png', cv2.IMREAD_GRAYSCALE)

gray_matrix = img.astype(float)
print(gray_matrix)

# Resize image
M, N = 50,50
X = cv2.resize(img, (N, M), interpolation=cv2.INTER_AREA)
plt.figure()
plt.imshow(X, cmap='gray')
plt.title('Resized Grayscale Image (Double Format)')

Xl = X.astype(float).reshape(M * N, 1)

# Gaussian parameters
s = 1.5
hsize = 9

# 1D Gaussian vector (like MATLAB's fspecial)
x = np.linspace(-hsize // 2, hsize // 2, hsize)
g = np.exp(-(x ** 2) / (2 * s ** 2))
g /= np.sum(g)

# Padding
g_padded_M = np.zeros(M)
g_padded_M[:hsize] = g
g_padded_N = np.zeros(N)
g_padded_N[:hsize] = g

# Circulant matrices
Ac = circulant(g_padded_M)
Ar = circulant(g_padded_N)

# Blurring matrix A
A = kron(Ac, Ar)

# Blurred image
B = A @ Xl

# Add noise
delta = 0.001
eta = delta * np.linalg.norm(B) * np.random.randn(*B.shape)
Bdel = B + eta

# Show blurred image
B1 = B.reshape(M, N)
plt.figure()
plt.imshow(B1, cmap='gray')
plt.title('Blurred Image (B)')

# Show noisy image
B1del = Bdel.reshape(M, N)
plt.figure()
plt.imshow(B1del, cmap='gray')
plt.title('Noisy Image (Bdel)')

# Iterative deblurring
X0 = np.zeros((M * N, 1))
q = 0.1
err = []

for i in range(1, 5):
    alpha = q ** i
    W = B - A @ X0
    F1 = np.linalg.inv(A + alpha * np.eye(M * N))
    F2 = alpha * F1 @ np.linalg.inv(A + (alpha ** 2) * np.eye(M * N))
    X1 = X0 + (F1 + F2) @ W
    err.append(np.linalg.norm(Xl - X1))
    X0 = X1

# Plot error
plt.figure()
plt.plot(err)
plt.title("Error")

# Show final deblurred image
X_es = X1.reshape(M, N)
plt.figure()
plt.imshow(X_es, cmap='gray')
plt.title('Estimated Image')
plt.show()
sys.exit()
