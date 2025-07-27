import os
import cv2
import numpy as np
import subprocess

# --- Configuration ---
input_folder = "raw_input/linear-gradient-patterns"
output_folder = "processed/processed_tiffs"
cropped_folder = "processed/roi_cropped_tiffs"
preview_folder = "processed/preview_8bit"
mask_path = "processed/face_mask_for_white_gradient.png"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(cropped_folder, exist_ok=True)
os.makedirs(preview_folder, exist_ok=True)

file_mapping = {
    "DSC07659.ARW": "x_pos.tiff",
    "DSC07660.ARW": "x_neg.tiff",
    "DSC07655.ARW": "y_pos.tiff",
    "DSC07656.ARW": "y_neg.tiff",
    "DSC07658.ARW": "z_pos.tiff",
    "DSC07657.ARW": "z_neg.tiff",
    "DSC07652.ARW": "constant.tiff"
}

bayer_pattern = cv2.COLOR_BayerRG2BGR

# Load face mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError("❌ Mask image not found or unreadable.")

bounding_rect = None
images_for_preview = []

# Step 1: Convert and save TIFFs
for arw_name, output_name in file_mapping.items():
    arw_path = os.path.join(input_folder, arw_name)
    pgm_path = arw_path.replace(".ARW", ".pgm")

    print(f"\n[1] Converting {arw_name} to Bayer PGM...")
    with open(pgm_path, "wb") as f:
        subprocess.run(["dcraw", "-D", "-4", "-c", arw_path], stdout=f)

    print(f"[2] Loading Bayer image: {pgm_path}")
    bayer = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
    if bayer is None:
        print(f"    ❌ Failed to load {pgm_path}")
        continue

    print(f"[3] Demosaicing...")
    bgr = cv2.demosaicing(bayer, bayer_pattern)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    output_path = os.path.join(output_folder, output_name)

    if "constant" in output_name:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(output_path, gray)
        rgb_path = output_path.replace(".tiff", "_rgb.tiff")
        cv2.imwrite(rgb_path, rgb)
        images_for_preview.append((output_name, gray))
        images_for_preview.append((os.path.basename(rgb_path), rgb))
        print(f"[4] Saved: {output_path} (grayscale), {rgb_path} (RGB)")
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(output_path, gray)
        images_for_preview.append((output_name, gray))
        print(f"[4] Saved grayscale: {output_path}")

    os.remove(pgm_path)

# Step 2: Compute bounding rect of the mask
if mask.shape != gray.shape:
    mask_resized = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
else:
    mask_resized = mask

x, y, w, h = cv2.boundingRect(mask_resized)
bounding_rect = (x, y, w, h)
cropped_mask = mask_resized[y:y+h, x:x+w]
cropped_mask_path = os.path.splitext(mask_path)[0] + "_cropped.png"
cv2.imwrite(cropped_mask_path, cropped_mask)
print(f"\n✅ Saved cropped mask: {cropped_mask_path}")

# Step 3: Save cropped versions + 8-bit previews
print("\n[5] Saving cropped versions and previews:")
for filename, image in images_for_preview:
    is_rgb = image.ndim == 3
    base = os.path.splitext(filename)[0]

    # Apply mask
    if is_rgb:
        masked = cv2.bitwise_and(image, image, mask=mask_resized[..., None])
        cropped = masked[y:y+h, x:x+w]
        norm_8bit = np.zeros_like(cropped, dtype=np.uint8)
        for c in range(3):
            channel = cropped[..., c]
            masked_pixels = channel[cropped_mask > 0]
            if masked_pixels.size > 0:
                min_val = np.percentile(masked_pixels, 1)
                max_val = np.percentile(masked_pixels, 99)
                norm = np.clip((channel - min_val) / (max_val - min_val) * 255, 0, 255)
                norm_8bit[..., c] = norm.astype(np.uint8)
    else:
        masked = cv2.bitwise_and(image, image, mask=mask_resized)
        cropped = masked[y:y+h, x:x+w]
        masked_pixels = cropped[cropped_mask > 0]
        min_val = np.percentile(masked_pixels, 1)
        max_val = np.percentile(masked_pixels, 99)
        norm_8bit = np.clip((cropped - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

    # Save cropped
    cropped_path = os.path.join(cropped_folder, base + ".tiff")
    cv2.imwrite(cropped_path, cropped)

    # Save preview
    preview_path = os.path.join(preview_folder, base + "_preview.png")
    cv2.imwrite(preview_path, norm_8bit)

    print(f"✅ {cropped_path}, 🖼️ {preview_path}")
