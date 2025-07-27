import cv2
import numpy as np
import os
from tqdm import tqdm

def stylized_shading(dot, tone_levels=[0.2, 0.6]):
    """
    Applies 3-tone quantized shading to simulate a stylized cartoon-like lighting.
    
    Parameters:
        dot (ndarray): Dot product of normal and light vectors per pixel.
        tone_levels (list): Threshold values to split tones into 3 bins.
        
    Returns:
        ndarray: Stylized shading map with discrete levels.
    """
    shaded = np.zeros_like(dot)
    shaded[dot < tone_levels[0]] = 0.2
    shaded[(dot >= tone_levels[0]) & (dot < tone_levels[1])] = 0.5
    shaded[dot >= tone_levels[1]] = 1.0
    return shaded

def run(
    normal_map_path,
    albedo_map_path,
    output_dir,
    fps=60,
    duration=20,
    intensity_scale=10,
    custom_filename=None,
    tone_levels=[0.2, 0.6],
    composite_background=False,
    background_path=None,
    mask_path=None,
):
    """
    Generates a stylized lighting animation using tone-quantized shading and horizontal light orbit.
    Optionally composites the shaded result with a background using a binary mask.

    Parameters:
        normal_map_path (str): Path to the normal map image.
        albedo_map_path (str): Path to the albedo map image.
        output_dir (str): Output directory for video.
        fps (int): Frames per second.
        duration (int): Duration in seconds.
        intensity_scale (float): Brightness multiplier for shading.
        custom_filename (str, optional): Optional output filename.
        tone_levels (list): Two thresholds for tone quantization.
        composite_background (bool): If True, blend shaded result with background using mask.
        background_path (str): Background image path (required if compositing).
        mask_path (str): Binary mask image path (required if compositing).
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and normalize maps ---
    normal_bgr = cv2.imread(normal_map_path)
    normal_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    albedo = cv2.imread(albedo_map_path).astype(np.float32) / 255.0

    # --- Convert normals to [-1, 1] space ---
    normals = normal_rgb * 2.0 - 1.0
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-8)

    H, W, _ = normals.shape
    n_frames = duration * fps

    # --- Optional compositing prep ---
    if composite_background:
        if background_path is None or mask_path is None:
            raise ValueError("To composite background, both background_path and mask_path must be provided.")
        background = cv2.imread(background_path).astype(np.float32) / 255.0
        background = cv2.resize(background, (W, H))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)[..., np.newaxis]

    # --- Light directions for 360° horizontal orbit ---
    angles = np.linspace(np.pi, -np.pi, n_frames)
    light_dirs = [np.array([np.cos(a), 0, np.sin(a)], dtype=np.float32) for a in angles]

    # --- Define output path ---
    filename = custom_filename if custom_filename else "stylized_diffuse.mp4"
    if not filename.lower().endswith(".mp4"):
        filename += ".mp4"
        
    out_path = os.path.join(output_dir, filename)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"\n🎥 Rendering: Stylized Quantized Diffuse {'(with background)' if composite_background else ''}")

    # --- Write animation frames ---
    for L in tqdm(light_dirs, desc=f"  → {filename}"):
        L = L / np.linalg.norm(L)
        dot = np.maximum(0, np.sum(normals * L, axis=2, keepdims=True))
        stylized = stylized_shading(dot, tone_levels=tone_levels) * intensity_scale
        shaded = albedo * stylized
        shaded = np.clip(shaded, 0, 1)

        if composite_background:
            frame = shaded * mask + background * (1 - mask)
        else:
            frame = shaded

        frame = (frame * 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"✅ Saved: {out_path}")
