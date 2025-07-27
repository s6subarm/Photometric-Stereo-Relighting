import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# --- Output folder ---
output_folder = "output/generated_albedos"
os.makedirs(output_folder, exist_ok=True)

# --- File chooser for albedo map ---
def select_albedo_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Albedo Map (PNG or JPG)",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tiff")],
        initialdir="output/photometric_stereo_results"
    )
    return file_path

# --- Prompt for custom filename ---
def prompt_filename(style_name, default_name):
    name = input(f"Filename for {style_name} (without .png) [default: {default_name}]: ").strip()
    return name if name else default_name

# --- Load selected albedo ---
input_albedo_path = select_albedo_file()
if not input_albedo_path:
    print("❗ No file selected. Exiting.")
    exit()

albedo = cv2.imread(input_albedo_path).astype(np.float32) / 255.0

# --- Style 1: Cobalt Bronze ---
def generate_cobalt_albedo(albedo):
    cobalt_tint = np.array([0.6, 0.45, 0.2])
    result = albedo * cobalt_tint
    return np.clip(result * 5, 0, 1)

# --- Style 2: Meta Black Bronze ---
def generate_meta_black_albedo(albedo):
    gray = cv2.cvtColor(albedo, cv2.COLOR_BGR2GRAY)
    shadow_boost = np.clip(1.2 - gray, 0, 1)
    shaded = albedo * shadow_boost[..., np.newaxis]
    tint = np.array([0.15, 0.13, 0.10])
    result = np.power(shaded * tint, 0.6)
    blurred = cv2.GaussianBlur(result, (0, 0), 1.2)
    result = cv2.addWeighted(result, 1.4, blurred, -0.4, 0)
    return np.clip(result, 0, 1)

# --- Style 3: Polished Silver ---
def generate_silver_albedo(albedo):
    gray = cv2.cvtColor(albedo, cv2.COLOR_BGR2GRAY)
    mono = cv2.merge([gray, gray, gray])
    boost = np.clip(1.3 - gray, 0, 1)
    shaded = mono * boost[..., np.newaxis]
    tint = np.array([0.85, 0.88, 0.95])
    result = np.power(shaded * tint, 0.5)
    blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
    result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
    return np.clip(result, 0, 1)

# --- Generate and save with custom names ---
cobalt_name = prompt_filename("Cobalt Bronze", "albedo_cobalt")
meta_black_name = prompt_filename("Meta Black Bronze", "albedo_meta_black")
silver_name = prompt_filename("Polished Silver", "albedo_silver")

cv2.imwrite(os.path.join(output_folder, cobalt_name + ".png"), (generate_cobalt_albedo(albedo) * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_folder, meta_black_name + ".png"), (generate_meta_black_albedo(albedo) * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_folder, silver_name + ".png"), (generate_silver_albedo(albedo) * 255).astype(np.uint8))

print("\n✅ Stylized albedo maps saved in:", output_folder)
