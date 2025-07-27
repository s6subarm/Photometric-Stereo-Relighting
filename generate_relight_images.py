import cv2
import numpy as np
import os
from tkinter import Tk, filedialog

# --- File Picker with fallback ---
def select_file_dialog(title, default_path):
    root = Tk()
    root.withdraw()
    initial_dir = os.path.dirname(default_path)
    file_path = filedialog.askopenfilename(title=title, initialdir=initial_dir)
    root.destroy()
    if not file_path:
        print(f"Nothing selected for '{title}', using default: {default_path}")
        return default_path
    return file_path

# --- Default paths ---
DEFAULT_NORMAL_MAP = "output/photometric_stereo_results/normal_map.png"
DEFAULT_ALBEDO_MAP = "output/generated_albedos/albedo_default.png"
DEFAULT_BACKGROUND = "processed/museum_background.jpg"
DEFAULT_MASK = "processed/face_mask_for_white_gradient_cropped.png"

# --- Ask to select normal map ---
use_custom_normal = input("Do you want to select a normal map? [y/n]: ").strip().lower() == "y"
normal_map_path = (
    select_file_dialog("Select Normal Map", DEFAULT_NORMAL_MAP)
    if use_custom_normal else DEFAULT_NORMAL_MAP
)
print(f"Using normal map: {normal_map_path}")

# --- Ask to select albedo map ---
use_custom_albedo = input("Do you want to select an albedo map? [y/n]: ").strip().lower() == "y"
albedo_path = (
    select_file_dialog("Select Albedo Map", DEFAULT_ALBEDO_MAP)
    if use_custom_albedo else DEFAULT_ALBEDO_MAP
)
print(f"Using albedo map: {albedo_path}")

# --- Ask whether to composite with background ---
composite_background = input("Do you want to add a background image? [y/n]: ").strip().lower() == "y"
if composite_background:
    background_path = select_file_dialog("Select Background", DEFAULT_BACKGROUND)
    mask_path = select_file_dialog("Select Binary Mask", DEFAULT_MASK)
    print(f"Using background: {background_path}")
    print(f"Using mask: {mask_path}")
else:
    background_path = None
    mask_path = None
    print("No background will be composited.")

# --- Prompt for intensity scale ---
intensity_input = input("Enter intensity scale (default = 10): ").strip()
intensity_scale = float(intensity_input) if intensity_input else 10.0
print(f"Using intensity scale: {intensity_scale}")

# --- Prompt for output suffix ---
custom_suffix = input("Enter custom filename suffix (e.g., rgb, metal, silver), or leave blank: ").strip()
print(f"Using custom suffix: '{custom_suffix}'")

# --- Output directory ---
output_dir = "output/lambertian_relighting_images"
os.makedirs(output_dir, exist_ok=True)

# --- Load and prepare normal map ---
normal_bgr = cv2.imread(normal_map_path)
normal_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
normals = normal_rgb * 2.0 - 1.0
norm = np.linalg.norm(normals, axis=2, keepdims=True)
normals = normals / (norm + 1e-8)
H, W = normals.shape[:2]

# --- Load albedo ---
albedo_bgr = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
if albedo_bgr.ndim == 2 or albedo_bgr.shape[2] == 1:
    gray = albedo_bgr.astype(np.float32) / 255.0
    albedo = np.stack([gray] * 3, axis=-1)
else:
    albedo = cv2.cvtColor(albedo_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# --- Prepare background and mask if needed ---
if composite_background:
    background = cv2.imread(background_path).astype(np.float32) / 255.0
    background = cv2.resize(background, (W, H))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.float32)[..., np.newaxis]

# --- Static lighting directions ---
# To use any other combinations custom light directions can be added.
# [+x,+y,+z] --> [right,top,front] light directions from viewer's POV. 
light_directions = {
    "frontal":              np.array([0, 0, 1], dtype=np.float32),
    #"frontal-top":          np.array([0, 1, 1], dtype=np.float32),
    #"frontal-bottom":       np.array([0, -1, 1], dtype=np.float32),
    #"frontal-left":         np.array([-1, 0, 1], dtype=np.float32),
    #"frontal-right":        np.array([1, 0, 1], dtype=np.float32),
    #"left":                 np.array([-1, 0, 0], dtype=np.float32),
    #"right":                np.array([1, 0, 0], dtype=np.float32),
    #"top":                  np.array([0, 1, 0], dtype=np.float32),
    #"top-right":            np.array([1, 1, 0], dtype=np.float32),
    #"top-left":             np.array([-1, 1, 0], dtype=np.float32),
    #"bottom":               np.array([0, -1, 0], dtype=np.float32),
    #"bottom-right":         np.array([1, -1, 0], dtype=np.float32),
    #"bottom-left":          np.array([-1, -1, 0], dtype=np.float32),
    #"front-top-left":       np.array([-1, 1, 1], dtype=np.float32),
    #"front-top-right":      np.array([1, 1, 1], dtype=np.float32),
    #"front-bottom-left":    np.array([-1, -1, 1], dtype=np.float32),
    #"front-bottom-right":   np.array([1, -1, 1], dtype=np.float32),
}

# --- Relighting loop ---
print("\nRendering relighting images:")
for name, light_dir in light_directions.items():
    L = light_dir / np.linalg.norm(light_dir)
    dot = np.maximum(0, np.sum(normals * L, axis=2, keepdims=True))
    shaded = albedo * dot * intensity_scale
    shaded = np.clip(shaded, 0, 1)

    if composite_background:
        final = shaded * mask + background * (1 - mask)
    else:
        final = shaded

    final_img = (final * 255).astype(np.uint8)
    final_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(output_dir, f"relit_{name}_{custom_suffix}.png")
    cv2.imwrite(out_path, final_bgr)
    print(f"✅ Saved: {out_path}")

print("\nAll relit images saved in:", output_dir)
