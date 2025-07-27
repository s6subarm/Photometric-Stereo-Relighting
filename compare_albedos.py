import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Paths ---
ps_albedo_gray_path = "output/photometric_stereo_results/albedo_map.png"
ps_albedo_rgb_path = "output/photometric_stereo_results/albedo_map_rgb.png"
true_albedo_gray_path = "processed/roi_cropped_tiffs/constant.tiff"
true_albedo_rgb_path = "processed/roi_cropped_tiffs/constant_rgb.tiff"
mask_path = "processed/face_mask_for_white_gradient_cropped.png"
output_image_path = "output/photometric_stereo_results/compare_albedos.png"

# --- Load grayscale albedos ---
ps_albedo_gray = cv2.imread(ps_albedo_gray_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
true_albedo_gray = cv2.imread(true_albedo_gray_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

# Normalize grayscale maps
ps_albedo_gray /= (ps_albedo_gray.max() + 1e-8)
true_albedo_gray /= (true_albedo_gray.max() + 1e-8)

# Resize for consistency
if ps_albedo_gray.shape != true_albedo_gray.shape:
    true_albedo_gray = cv2.resize(true_albedo_gray, (ps_albedo_gray.shape[1], ps_albedo_gray.shape[0]))

# --- Load RGB photometric stereo albedo ---
ps_albedo_rgb = cv2.imread(ps_albedo_rgb_path, cv2.IMREAD_UNCHANGED)
ps_albedo_rgb = cv2.cvtColor(ps_albedo_rgb, cv2.COLOR_BGR2RGB)

# --- Generate true RGB albedo from constant illumination image ---
true_rgb = cv2.imread(true_albedo_rgb_path, cv2.IMREAD_UNCHANGED)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if true_rgb is None or mask is None:
    raise FileNotFoundError("Missing constant_rgb.tiff or roi_mask_cropped.png")

# Normalize true RGB albedo using percentiles within the mask
norm_rgb = np.zeros_like(true_rgb, dtype=np.uint8)
for c in range(3):
    channel = true_rgb[..., c].astype(np.float32)
    masked = channel[mask > 0]
    if masked.size > 0:
        p1, p99 = np.percentile(masked, 1), np.percentile(masked, 99)
        norm = np.clip((channel - p1) / (p99 - p1) * 255, 0, 255)
        norm_rgb[..., c] = norm.astype(np.uint8)
true_albedo_rgb = cv2.cvtColor(norm_rgb, cv2.COLOR_BGR2RGB)

# Resize all RGB to match grayscale if needed
target_shape = (ps_albedo_gray.shape[1], ps_albedo_gray.shape[0])
ps_albedo_rgb = cv2.resize(ps_albedo_rgb, target_shape)
true_albedo_rgb = cv2.resize(true_albedo_rgb, target_shape)

# --- Plot in 1x4 grid ---
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(ps_albedo_gray, cmap='gray')
axs[0].set_title("PS Albedo (Grayscale)")
axs[0].axis('off')

axs[1].imshow(true_albedo_gray, cmap='gray')
axs[1].set_title("True Albedo (Grayscale)")
axs[1].axis('off')

axs[2].imshow(ps_albedo_rgb)
axs[2].set_title("PS Albedo (RGB)")
axs[2].axis('off')

axs[3].imshow(true_albedo_rgb)
axs[3].set_title("True Albedo (RGB)")
axs[3].axis('off')

plt.tight_layout()
plt.savefig(output_image_path, dpi=300)
plt.show()

print(f"✅ Albedo grid saved to: {output_image_path}")
