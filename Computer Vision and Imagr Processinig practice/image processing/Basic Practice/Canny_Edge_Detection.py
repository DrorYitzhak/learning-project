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



# edges_canny = cv2.Canny(f, 200, 230)
# display(edges_canny, "Canny Edge Detection 150, 200")

edges_canny = cv2.Canny(f, 150, 200)
display(edges_canny, "Canny Edge Detection 150, 200")
#
# edges_canny = cv2.Canny(f, 100, 150)
# display(edges_canny, "Canny Edge Detection 100, 150")
#
# edges_canny = cv2.Canny(f, 50, 100)
# display(edges_canny, "Canny Edge Detection 50, 100")



# Marr-Hildreth (LoG) edge detection
edges_marr_hildreth = cv2.Laplacian(cv2.GaussianBlur(f, (3, 3), 5), cv2.CV_64F)
edges_marr_hildreth = np.uint8(np.absolute(edges_marr_hildreth))
display(edges_marr_hildreth, "Marr-Hildreth (LoG) Edge Detection")
display(edges_canny, "Canny Edge Detection 150, 200")

plt.show()
print("o")