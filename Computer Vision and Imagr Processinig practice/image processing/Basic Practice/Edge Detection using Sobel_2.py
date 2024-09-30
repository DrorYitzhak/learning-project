import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Path to the image
image_path = "\\PycharmProjects\\WhatsApp Image 2023-11-22 at 21.40.58_5d373176.jpg"

# Read the original image
f = cv2.imread(image_path, 0)

# function for displaying image
def display(img, title):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(title)
# sobel kernel
sobel_x = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

sobel_y = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# prewitt kernel
#
# sobel_x = np.array([[-1, -1, -1],
#                     [ 0,  0,  0],
#                     [ 1,  1,  1]])
#
# sobel_y = np.array([[-1, 0, 1],
#                     [-1, 0, 1],
#                     [-1, 0, 1]])


# partial derivative in x-direction
edge_x = cv2.filter2D(src=f, ddepth=-1, kernel=sobel_x)
display(edge_x, "X_Edge Detection")

# partial derivative in y-direction
edge_y = cv2.filter2D(src=f, ddepth=-1, kernel=sobel_y)
display(edge_y, "Y_Edge Detection")

add_edge = edge_y + edge_x
ret, thresholded_edge_x = cv2.threshold(add_edge, 70, 120, cv2.THRESH_BINARY)
display(add_edge, "X+Y_Edge Detection")
display(thresholded_edge_x, "Edge Detection")

edges_canny = cv2.Canny(f, 150, 200)
display(edges_canny, "Canny Edge Detection")

plt.show()
print("o")