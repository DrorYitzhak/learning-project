import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the image
image_path = "C:\\PycharmProjects\\Image_2.jpg"

# Read the original image
f = cv2.imread(image_path, 0)

#  ========================================= Perform 2D Fourier Transform ==========================================
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

# Create Gaussian Filter: Low Pass Filter
M, N = f.shape
H = np.zeros((M, N), dtype=np.float32)
D0 = 10
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
        H[u, v] = np.exp(-D**2 / (2 * D0**2))

# Create Gaussian: High pass filter
HPF = 1 - H

# Apply the low pass filter
Gshift_low_pass = Fshift * H
G_low_pass = np.fft.ifftshift(Gshift_low_pass)
g_low_pass = np.abs(np.fft.ifft2(G_low_pass))

# Apply the high pass filter
Gshift_high_pass = Fshift * HPF
G_high_pass = np.fft.ifftshift(Gshift_high_pass)
g_high_pass = np.abs(np.fft.ifft2(G_high_pass))

# Display images side by side
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

# Original Image
axs[0, 0].imshow(f, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# Magnitude Spectrum
axs[0, 1].imshow(np.log1p(np.abs(Fshift)), cmap='gray')
axs[0, 1].set_title('Magnitude Spectrum')
axs[0, 1].axis('off')

# Low Pass Filter
axs[0, 2].imshow(H, cmap='gray')
axs[0, 2].set_title('Low Pass Filter (Gaussian)')
axs[0, 2].axis('off')

# Image after Low Pass Filtering
axs[0, 3].imshow(g_low_pass, cmap='gray')
axs[0, 3].set_title('Image after Low Pass Filtering')
axs[0, 3].axis('off')

# High Pass Filter
axs[1, 0].imshow(HPF, cmap='gray')
axs[1, 0].set_title('High Pass Filter')
axs[1, 0].axis('off')

# Image after High Pass Filtering
axs[1, 1].imshow(g_high_pass, cmap='gray')
axs[1, 1].set_title('Image after High Pass Filtering')
axs[1, 1].axis('off')

# Magnitude Spectrum after Low Pass Filtering
axs[1, 2].imshow(np.log1p(np.abs(Gshift_low_pass)), cmap='gray')
axs[1, 2].set_title('Magnitude Spectrum after Low Pass Filtering')
axs[1, 2].axis('off')

# Magnitude Spectrum after High Pass Filtering
axs[1, 3].imshow(np.log1p(np.abs(Gshift_high_pass)), cmap='gray')
axs[1, 3].set_title('Magnitude Spectrum after High Pass Filtering')
axs[1, 3].axis('off')

cv2.filter2D()

plt.tight_layout()
plt.show()
print("h")