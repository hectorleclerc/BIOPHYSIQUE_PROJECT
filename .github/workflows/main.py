import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# --- 1. Charger l'image ---
img = cv2.imread(r"C:\Users\hecto\Documents\ENS\M2\Biophysique/champignon_635ma.jpg", cv2.IMREAD_GRAYSCALE)

# --- 2. Prétraitement ---
# Lissage
img_blur = cv2.GaussianBlur(img, (5,5), 0)

# Binarisation (adaptative)
th = cv2.adaptiveThreshold(
    img_blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    11, 3
)

# --- 3. Squelettisation ---
skeleton = skeletonize(th // 255).astype(np.uint8)

# --- 4. Plot ---
plt.figure(figsize=(14,5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Binarized")
plt.imshow(th, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Skeleton")
plt.imshow(skeleton, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
