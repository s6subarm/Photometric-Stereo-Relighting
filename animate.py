import os
import tkinter as tk
from tkinter import filedialog
from animate_standard_diffused_lambertian import run as run_lambertian
from animate_stylized_quantized_diffuse import run as run_stylized
from animate_phong_metalic import run as run_phong_metalic

# --- User-friendly input helpers ---
def prompt_bool(message):
    return input(f"{message} [y/n]: ").strip().lower() == 'y'

def prompt_int(message, default):
    value = input(f"{message} (default: {default}): ").strip()
    return int(value) if value else default

def prompt_float(message, default):
    value = input(f"{message} (default: {default}): ").strip()
    return float(value) if value else default

def prompt_string(message):
    return input(f"{message} (leave empty for default): ").strip()

# --- File selection helper with optional popup ---
def select_custom_file(default_path, prompt_message=None, filetypes=[("PNG images", "*.png")], initialdir="."):
    if prompt_message:
        use_custom = prompt_bool(prompt_message)
        if not use_custom:
            return default_path
    try:
        root = tk.Tk()
        root.withdraw()
        selected_file = filedialog.askopenfilename(
            initialdir=initialdir,
            title="Select File",
            filetypes=filetypes
        )
        root.destroy()
        if selected_file:
            return selected_file
        else:
            print("No file selected. Using default.")
    except Exception as e:
        print("Popup failed:", e)
    return default_path

# --- Main Controller ---
def main():
    print(" Animation Generator\n")

    # --- Selection Phase ---
    options = {
        "standard": prompt_bool("1. Generate Standard Diffused Lambertian?"),
        "stylized": prompt_bool("2. Generate Stylized Quantized Diffuse?"),
        "phong_metalic": prompt_bool("3. Generate Phong-Based Metallic Animation?"),
    }

    # --- Shared parameters ---
    print("\n Base Animation Settings:")
    fps = prompt_int("Frames per second", 60)
    duration = prompt_int("Duration in seconds", 20)

    tasks = []

    # --- Shared normal map selection ---
    normal_map_path = select_custom_file(
        default_path="output/photometric_stereo_results/normal_map.png",
        prompt_message="Do you want to select a custom normal map?",
        filetypes=[("PNG images", "*.png")],
        initialdir="output/photometric_stereo_results"
    )

    if options["standard"]:
        print("\n[Standard Lambertian Parameters]")
        albedo_path = select_custom_file(
            default_path="output/photometric_stereo_results/albedo_map_rgb.png",
            prompt_message="Do you want to select a custom albedo file?",
            filetypes=[("PNG images", "*.png")],
            initialdir="output/generated_albedos"
        )

        composite_bg = prompt_bool("Do you want to composite a background?")
        if composite_bg:
            print("→ Choose background and mask for compositing...")
            background_path = select_custom_file(
                default_path="processed/museum_background.jpg",
                prompt_message=None,
                filetypes=[("Image files", "*.png *.jpg *.jpeg")],
                initialdir="processed"
            )
            mask_path = select_custom_file(
                default_path="processed/face_mask_for_white_gradient_cropped.png",
                prompt_message=None,
                filetypes=[("PNG images", "*.png")],
                initialdir="processed"
            )
        else:
            background_path = None
            mask_path = None

        intensity = prompt_float("Brightness intensity scale", 10)
        custom_name = prompt_string("Custom filename (e.g., lambertian)")
        tasks.append({
            "func": run_lambertian,
            "kwargs": {
                "normal_map_path": normal_map_path,
                "albedo_map_path": albedo_path,
                "output_dir": "output/animations",
                "fps": fps,
                "duration": duration,
                "intensity_scale": intensity,
                "custom_filename": custom_name,
                "composite_background": composite_bg,
                "background_path": background_path,
                "mask_path": mask_path,
            }
        })

    if options["stylized"]:
        print("\n[Stylized Quantized Diffuse Parameters]")
        albedo_path = select_custom_file(
            default_path="output/photometric_stereo_results/albedo_map.png",
            prompt_message="Do you want to select a custom albedo file?",
            filetypes=[("PNG images", "*.png")],
            initialdir="output/generated_albedos"
        )

        composite_bg = prompt_bool("Do you want to composite a background?")
        if composite_bg:
            print("→ Choose background and mask for compositing...")
            background_path = select_custom_file(
                default_path="processed/museum_background.jpg",
                prompt_message=None,
                filetypes=[("Image files", "*.png *.jpg *.jpeg")],
                initialdir="processed"
            )
            mask_path = select_custom_file(
                default_path="processed/face_mask_for_white_gradient_cropped.png",
                prompt_message=None,
                filetypes=[("PNG images", "*.png")],
                initialdir="processed"
            )
        else:
            background_path = None
            mask_path = None

        intensity = prompt_float("Brightness intensity scale", 10)
        custom_name = prompt_string("Custom filename (e.g., stylized)")
        tasks.append({
            "func": run_stylized,
            "kwargs": {
                "normal_map_path": normal_map_path,
                "albedo_map_path": albedo_path,
                "output_dir": "output/animations",
                "fps": fps,
                "duration": duration,
                "intensity_scale": intensity,
                "custom_filename": custom_name,
                "composite_background": composite_bg,
                "background_path": background_path,
                "mask_path": mask_path,
            }
        })

    if options["phong_metalic"]:
        print("\n[Phong-Based Metallic Parameters]")
        albedo_path = select_custom_file(
            default_path="output/generated_albedos/albedo_cobalt.png",
            prompt_message="Do you want to select a custom albedo file?",
            filetypes=[("PNG images", "*.png")],
            initialdir="output/generated_albedos"
        )

        composite_bg = prompt_bool("Do you want to composite a background?")
        if composite_bg:
            print("→ Choose background and mask for compositing...")
            background_path = select_custom_file(
                default_path="processed/museum_background.jpg",
                prompt_message=None,
                filetypes=[("Image files", "*.png *.jpg *.jpeg")],
                initialdir="processed"
            )
            mask_path = select_custom_file(
                default_path="processed/face_mask_for_white_gradient_cropped.png",
                prompt_message=None,
                filetypes=[("PNG images", "*.png")],
                initialdir="processed"
            )
        else:
            background_path = None
            mask_path = None

        intensity = prompt_float("Brightness intensity scale", 2)
        top_light_strength = prompt_float("Top light vignette strength", 0.6)
        shininess = prompt_int("Specular shininess", 30)
        custom_name = prompt_string("Custom filename (e.g., phong_metallic)")

        tasks.append({
            "func": run_phong_metalic,
            "kwargs": {
                "normal_map_path": normal_map_path,
                "albedo_map_path": albedo_path,
                "output_dir": "output/animations",
                "fps": fps,
                "duration": duration,
                "intensity_scale": intensity,
                "top_light_strength": top_light_strength,
                "shininess": shininess,
                "custom_filename": custom_name,
                "composite_background": composite_bg,
                "background_path": background_path,
                "mask_path": mask_path,
            }
        })

    # --- Execution Phase ---
    for task in tasks:
        print("\n Starting animation...")
        task["func"](**task["kwargs"])

    print("\n✅ All selected animations generated successfully.")

if __name__ == "__main__":
    main()
