import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

from scipy.linalg import circulant, kron

# Load and convert RGB image
img = cv2.imread('nargb.jpeg')   # Load image in color (default)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Resize image
M, N = 100, 100
img_resized = cv2.resize(img, (N, M), interpolation=cv2.INTER_AREA)

plt.figure()
plt.imshow(img_resized)
plt.title('Resized RGB Image')

# Gaussian parameters
s = 1.5
hsize = 9

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

# Prepare outputs
blurred_channels = []
noisy_channels = []
deblurred_channels = []
errors = []

q = 0.1

# Process each RGB channel separately
for ch in range(3):
    channel = img_resized[:, :, ch].astype(float)
    Xl = channel.reshape(M * N, 1)

    # Blur
    B = A @ Xl

    # Add noise
    delta = 0.001
    eta = delta * np.linalg.norm(B) * np.random.randn(*B.shape)
    Bdel = B + eta

    # Iterative deblurring
    X0 = np.zeros((M * N, 1))
    err = []

    for i in range(1, 5):
        alpha = q ** i
        W = B - A @ X0
        F1 = np.linalg.inv(A + alpha * np.eye(M * N))
        F2 = alpha * F1 @ np.linalg.inv(A + (alpha ** 2) * np.eye(M * N))
        X1 = X0 + (F1 + F2) @ W
        err.append(np.linalg.norm(Xl - X1))
        X0 = X1

    errors.append(err)
    blurred_channels.append(B.reshape(M, N))
    noisy_channels.append(Bdel.reshape(M, N))
    deblurred_channels.append(X1.reshape(M, N))

# Stack and clip results
blurred_img = np.stack(blurred_channels, axis=2).clip(0, 255).astype(np.uint8)
noisy_img = np.stack(noisy_channels, axis=2).clip(0, 255).astype(np.uint8)
deblurred_img = np.stack(deblurred_channels, axis=2).clip(0, 255).astype(np.uint8)

# Display results
plt.figure()
plt.imshow(blurred_img)
plt.title('Blurred Image')

plt.figure()
plt.imshow(noisy_img)
plt.title('Noisy Image')

plt.figure()
plt.imshow(deblurred_img)
plt.title('Estimated (Deblurred) Image')

# Plot error for R channel
plt.figure()
plt.plot(errors[0])
plt.title('Error (Red Channel)')
plt.show()

sys.exit()
