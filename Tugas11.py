import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

# 1. Load the input image (grayscale)
image_path = "images.jpeg"  # Replace with your image path
image = iio.imread(image_path, mode="L")

# 2. Apply Sobel filter for edge detection
sobel_x = sobel(image, axis=0)  # Sobel filter in the x-direction
sobel_y = sobel(image, axis=1)  # Sobel filter in the y-direction
sobel_magnitude = np.hypot(sobel_x, sobel_y)  # Compute the gradient magnitude
sobel_magnitude = (sobel_magnitude / sobel_magnitude.max() * 255).astype(np.uint8)

# 3. Perform basic thresholding
threshold = 128  # Define a threshold value
binary_image = sobel_magnitude > threshold  # Apply thresholding
binary_image = binary_image.astype(np.uint8) * 255  # Convert to binary (0 or 255)

# 4. Plot results
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Sobel Edge Detection")
plt.imshow(sobel_magnitude, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Thresholded Image")
plt.imshow(binary_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
