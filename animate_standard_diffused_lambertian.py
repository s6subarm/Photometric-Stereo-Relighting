import cv2
import numpy as np
import os
from tqdm import tqdm

def run(
    normal_map_path,
    albedo_map_path,
    output_dir,
    fps=60,
    duration=20,
    intensity_scale=10,
    custom_filename=None,
    composite_background=False,
    background_path=None,
    mask_path=None,
):
    """
    Generates an animation using basic Lambertian shading with a horizontally orbiting light source.
    Optionally composites the output with a background image using a mask.

    Parameters:
        normal_map_path (str): Path to the normal map image.
        albedo_map_path (str): Path to the albedo map image.
        output_dir (str): Directory where the video will be saved.
        fps (int): Frames per second of the output video.
        duration (int): Duration of the animation in seconds.
        intensity_scale (float): Multiplier for shading brightness.
        custom_filename (str, optional): Custom filename for the output video.
        composite_background (bool): If True, composites shaded result with background using mask.
        background_path (str): Path to background image (required if compositing).
        mask_path (str): Path to binary face mask (required if compositing).
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and normalize input maps ---
    normal_bgr = cv2.imread(normal_map_path)
    normal_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    albedo = cv2.imread(albedo_map_path).astype(np.float32) / 255.0

    # --- Convert normals from RGB [0,1] to XYZ [-1,1] and normalize ---
    normals = normal_rgb * 2.0 - 1.0
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-8)

    H, W, _ = normals.shape
    n_frames = duration * fps

    # --- Optional compositing prep ---
    if composite_background:
        if background_path is None or mask_path is None:
            raise ValueError("To composite background, provide both background_path and mask_path.")
        background = cv2.imread(background_path).astype(np.float32) / 255.0
        background = cv2.resize(background, (W, H))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)[..., np.newaxis]

    # --- Simulate light moving in a horizontal 360° orbit around the subject ---
    angles = np.linspace(np.pi, -np.pi, n_frames)
    light_dirs = [np.array([np.cos(a), 0, np.sin(a)], dtype=np.float32) for a in angles]

    # --- Define output filename ---
    filename = custom_filename if custom_filename else "lambertian_standard_diffuse.mp4"
    if not filename.lower().endswith(".mp4"):
        filename += ".mp4"
        
    out_path = os.path.join(output_dir, filename)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"\n🎥 Rendering: Standard Diffused Lambertian {'(with background)' if composite_background else ''}")
    
    # --- Generate and write each frame ---
    for L in tqdm(light_dirs, desc=f"  → {filename}"):
        L = L / np.linalg.norm(L)
        dot = np.maximum(0, np.sum(normals * L, axis=2, keepdims=True))
        shaded = albedo * dot * intensity_scale
        shaded = np.clip(shaded, 0, 1)

        if composite_background:
            frame = shaded * mask + background * (1 - mask)
        else:
            frame = shaded

        frame = (frame * 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"✅ Saved: {out_path}")
