import cv2
from IPython.display import display, Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Path to the image
image_path = "C:\PycharmProjects\Image_2.jpg"

# Read the original image
original_image = cv2.imread(image_path)

#  =============================================== Noise =============================================================
#
# # Calculate mean and standard deviation of the image
mean_value = np.mean(original_image)
std_deviation = np.std(original_image)
image_size = original_image.shape
#
# # Generate noise for the image
noise = np.zeros(image_size, dtype=np.uint8)
noise_image = cv2.randn(noise, mean_value, std_deviation)
noisy_image = cv2.add(original_image, noise_image)
noisy_image2 = 10 + original_image

#
f = original_image/255
y = f.shape
mean = 0
var = 0.01
sigma = np.sqrt(var)
n = np.random.normal(loc=mean, scale=sigma, size=y)

# g = f + n
g = np.clip(f + n, 0, 1)

# # display all
cv2.imshow('original image', f)
cv2.imshow('Gaussian noise', n)
cv2.imshow('Corrupted Image', g)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
#


"""**Salt-and-Pepper Noise:**
- It's a type of image noise that appears as isolated pixels with either very high or very low intensity.
- "Salt" refers to bright pixels, and "pepper" refers to dark pixels.

**Median Filter:**
- A filtering technique used for noise removal.
- It replaces each pixel's intensity with the median value of the intensities in its neighborhood.
- Effective for salt-and-pepper noise as it identifies and removes extreme pixel values.

**Kernel Size:**
- The size of the neighborhood considered for calculating the median.
- Larger kernel sizes consider more pixels, providing better noise removal but may result in loss of image detail.

**Applying the Filter:**
1. **Read Noisy Image:** Load the image with salt-and-pepper noise.
2. **Apply Median Filter:** Use the `cv2.medianBlur` function with a specified kernel size.
3. **Display Images:** Show the original noisy image and the image after filtering.
"""


#  ======================================= Histogram Equalization ====================================================


# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization to the grayscale image
equalized_image = cv2.equalizeHist(gray_image)

# Calculate the histogram and cumulative distribution function (CDF) for the original grayscale image
hist_values, bin_edges, _ = plt.hist(gray_image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.5)
result_of_image = np.cumsum(hist_values, dtype=int)
res = result_of_image / max(result_of_image)

# Calculate the histogram and CDF for the equalized grayscale image
hist_values_2, bin_edges_2, _ = plt.hist(equalized_image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.5)
result_2 = np.cumsum(hist_values_2, dtype=int)
res_2 = result_2 / max(result_2)

# Create a grid layout
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])

# Display the regular grayscale image, its histogram, and CDF
ax1 = plt.subplot(gs[0])
ax1.imshow(gray_image, cmap='gray')
ax1.set_title("Regular Image")

ax2 = plt.subplot(gs[1])
ax2.hist(gray_image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.5)
ax2.set_title("Histogram Regular Image")

ax3 = plt.subplot(gs[2])
ax3.plot(np.arange(256), res, color='blue')
ax3.set_title("CDF Regular Image")

# Display the equalized grayscale image, its histogram, and CDF
ax4 = plt.subplot(gs[3])
ax4.imshow(equalized_image, cmap='gray')
ax4.set_title("Equalized Image")

ax5 = plt.subplot(gs[4])
ax5.hist(equalized_image.ravel(), bins=50, range=[0, 256], color='gray', alpha=0.7)
ax5.set_title("Histogram Equalized Image")

ax6 = plt.subplot(gs[5])
ax6.plot(np.arange(256), res_2, color='blue')
ax6.set_title("CDF Equalized Image")

# Adjust layout to prevent overlapping
plt.tight_layout()


# Show the plot
plt.show()


# ============================================= Average Filter =======================================================

n = 5
m = 5
denoise_average = cv2.blur(g, (n, m))
cv2.imshow("Noisy Image", g)
cv2.imshow("Denoised Image (Average Filter)", denoise_average)
cv2.imshow("original_image", original_image)

# ============================================= HPF fo example=======================================================

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_image", gray_image)
n = 5
m = 5
denoise_average = cv2.blur(gray_image, (n, m))
cv2.imshow("Denoised Image (Average Filter)", denoise_average)
high_pass_image = gray_image - 1.2*denoise_average
cv2.imshow('High_Pass_Image', high_pass_image)

# ============================================= Median Filter ========================================================
g = (g * 255).astype(np.uint8)
median_filter_size = 5
denoise_median = cv2.medianBlur(g, median_filter_size)
denoise_median = denoise_median / 255.0
cv2.imshow("Noisy Image", g)
cv2.imshow("Denoised Image (Median Filter)", denoise_median)
cv2.imshow("original_image", original_image)


# ============================================= Gaussian Blur Filter =================================================


blurred_image = cv2.GaussianBlur(g, (5, 5), 0)
cv2.imshow('Gaussian Blur Filter', blurred_image)

blurred_image = cv2.GaussianBlur(g, (5, 5), 2)
cv2.imshow('Gaussian Blur Filter 2', blurred_image)


blurred_image = cv2.GaussianBlur(g, (5, 5), 10)
cv2.imshow('Gaussian Blur Filter 3', blurred_image)

blurred_image = cv2.GaussianBlur(g, (3, 3), 2)
cv2.imshow('Gaussian Blur Filter 4', blurred_image)


high_pass_image = g - blurred_image
cv2.imshow('High_Pass_Image', high_pass_image)


sharp_image = g + (0.1)*(g + high_pass_image)
cv2.imshow('Sharp_Image', sharp_image)



# ============================================= Bilateral Filter =====================================================
#
# """Bilateral Filtering:
#
# Definition:
# Bilateral filtering is a non-linear, edge-preserving smoothing filter that averages pixels based on both spatial closeness and intensity similarity.
#
# Why Use It:
#
# Effective for smoothing images while preserving edges.
# Reduces noise without significant loss of image details.
# Particularly useful for images with sharp transitions and fine structures.
#
# src: Input image.
# d: Diameter of each pixel neighborhood. Influences spatial closeness.
# sigmaColor: Filter sigma in the color space. Controls intensity similarity.
# sigmaSpace: Filter sigma in the coordinate space. Influences spatial closeness.
# Choosing Parameters:
#
# d: Typically set to 2 times the sigmaSpace.
# sigmaColor and sigmaSpace: Higher values preserve more details but increase processing time. Adjust based on the level of noise and desired smoothing.
# Applicability to Noise Types:
# Recommended for:
#
# Salt-and-pepper noise.
# Gaussian noise.
# Poisson noise.
# Not Recommended for:
#
# Speckle noise."""
#
g = (g * 255).astype(np.uint8)
g_8bit = cv2.convertScaleAbs(g)
bilateral_filtered_image = cv2.bilateralFilter(g_8bit, d=9, sigmaColor=70, sigmaSpace=70)
cv2.imshow("Original Image", g_8bit)
cv2.imshow("Bilateral Filtered Image", bilateral_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




# ============================================= DFT2 =====================================================

# Read the original image
f = cv2.imread(image_path, 0)

F = np.fft.fft2(f)

plt.imshow(np.log1p(np.abs(F)), cmap='gray')












cv2.waitKey(0)
cv2.destroyAllWindows()




