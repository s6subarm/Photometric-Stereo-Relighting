import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread("processed/processed_tiffs/x_pos.tiff", cv2.IMREAD_UNCHANGED)

# Normalize for display
norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.imshow(norm_img, cmap='gray')
plt.title("x_pos.tiff (rescaled)")
plt.axis('off')
plt.show()


orig = cv2.imread("raw_input/linear-gradient-patterns/DSC07659.JPG", cv2.IMREAD_GRAYSCALE)
linear = cv2.imread("processed_tiffs/x_pos.tiff", cv2.IMREAD_UNCHANGED)
linear_norm = cv2.normalize(linear, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.subplot(1, 2, 1)
plt.imshow(orig, cmap='gray')
plt.title("Original JPEG")

plt.subplot(1, 2, 2)
plt.imshow(linear_norm, cmap='gray')
plt.title("Linear TIFF")

plt.show()


import matplotlib.pyplot as plt

plt.hist(img.ravel(), bins=256, range=(0, 65535))
plt.title("Histogram of 16-bit TIFF")
plt.show()
