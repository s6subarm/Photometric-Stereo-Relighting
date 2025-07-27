import os
import cv2
import numpy as np
from tqdm import tqdm

# --- Configuration ---
input_folder = "processed/roi_cropped_tiffs"
output_normal = "output/photometric_stereo_results/normal_map.png"
output_albedo_gray = "output/photometric_stereo_results/albedo_map.png"
output_albedo_rgb = "output/photometric_stereo_results/albedo_map_rgb.png"
constant_rgb_path = os.path.join(input_folder, "constant_rgb.tiff")
cropped_mask_path = "processed/face_mask_for_white_gradient_cropped.png"

# --- Light directions viewer's perspective ---
light_dirs = np.array([
    [1,  0,  0],   # x_pos
    [-1, 0,  0],   # x_neg
    [0,  1,  0],   # y_pos
    [0, -1,  0],   # y_neg
    [0,  0,  1],   # z_pos
    [0,  0, -1],   # z_neg
], dtype=np.float32)

file_order = [
    "x_pos.tiff", "x_neg.tiff",
    "y_pos.tiff", "y_neg.tiff",
    "z_pos.tiff", "z_neg.tiff"
]

# --- Load directional images ---
images = []
for filename in file_order:
    path = os.path.join(input_folder, filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    images.append(img)
images = np.stack(images, axis=-1)  # shape: (H, W, 6)

H, W, K = images.shape
M = light_dirs  # shape: (6, 3)

# --- Estimate normals and grayscale albedo ---
G = np.zeros((H, W, 3), dtype=np.float32)
albedo = np.zeros((H, W), dtype=np.float32)

for y in tqdm(range(H), desc="Solving for normals"):
    for x in range(W):
        I = images[y, x, :]
        if np.all(I == 0):
            continue
        G_vec, _, _, _ = np.linalg.lstsq(M, I, rcond=None)
        G[y, x, :] = G_vec
        albedo[y, x] = np.linalg.norm(G_vec)

# Normalize to unit normals
normals = np.zeros_like(G)
nonzero = albedo > 1e-5
normals[nonzero] = G[nonzero] / albedo[nonzero][..., np.newaxis]

# Save normal map
normal_map = ((normals + 1) / 2 * 255).astype(np.uint8)
cv2.imwrite(output_normal, cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))

# Save grayscale albedo map
albedo_img = cv2.normalize(albedo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(output_albedo_gray, albedo_img)

# --- RGB Albedo from constant illumination (normalized and scaled by grayscale albedo) ---
print("\n📸 Using constant illumination image directly as RGB albedo...")
rgb_ref = cv2.imread(constant_rgb_path, cv2.IMREAD_UNCHANGED)
if rgb_ref is None:
    raise FileNotFoundError(f"❌ Failed to load {constant_rgb_path}")

mask = cv2.imread(cropped_mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"❌ Failed to load {cropped_mask_path}")

# Normalize each channel based on masked pixels
norm_rgb = np.zeros_like(rgb_ref, dtype=np.uint8)
for c in range(3):
    channel = rgb_ref[..., c].astype(np.float32)
    masked_pixels = channel[mask > 0]
    if masked_pixels.size > 0:
        min_val = np.percentile(masked_pixels, 1)
        max_val = np.percentile(masked_pixels, 99)
        norm = np.clip((channel - min_val) / (max_val - min_val) * 255, 0, 255)
        norm_rgb[..., c] = norm.astype(np.uint8)

# Match intensity with grayscale albedo
gray_albedo_norm = albedo_img.astype(np.float32) / 255.0
rgb_scaled = norm_rgb.astype(np.float32) / 255.0
rgb_final = rgb_scaled * gray_albedo_norm[..., np.newaxis]
rgb_final = np.clip(rgb_final * 255, 0, 255).astype(np.uint8)
cv2.imwrite(output_albedo_rgb, rgb_final)

print(f"✅ Saved: {output_normal}, {output_albedo_gray}, {output_albedo_rgb}")
print("Normal map shape:", normal_map.shape)
print("Albedo gray shape:", albedo_img.shape)
print("Albedo RGB shape:", rgb_final.shape)
