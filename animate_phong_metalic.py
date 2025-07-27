import cv2
import numpy as np
import os
from tqdm import tqdm

def phong_metal_shading(normals, light_dir, view_dir, albedo, shininess=128):
    """Stable Phong-based metallic shading."""
    H, W, _ = normals.shape

    L = light_dir / np.linalg.norm(light_dir)
    V = view_dir / np.linalg.norm(view_dir)
    L_img = np.tile(L, (H, W, 1))
    V_img = np.tile(V, (H, W, 1))

    dotNL = np.clip(np.sum(normals * L_img, axis=2, keepdims=True), 0, 1)
    R = 2 * dotNL * normals - L_img
    R = R / (np.linalg.norm(R, axis=2, keepdims=True) + 1e-8)

    dotRV = np.clip(np.sum(R * V_img, axis=2, keepdims=True), 0, 1)
    visible = (dotNL > 0).astype(np.float32)
    specular = np.power(dotRV, shininess) * visible

    specular_tint = np.array([1.0, 0.95, 0.7])
    specular_colored = specular * specular_tint

    kd = 0.6
    ks = 1.0
    diffuse = kd * albedo * dotNL
    shaded = diffuse + ks * specular_colored
    shaded *= 1.5
    return np.clip(shaded, 0, 1)


def run(
    normal_map_path,
    albedo_map_path,
    output_dir,
    fps=15,
    duration=3,
    intensity_scale=1.5,
    top_light_strength=0.2,
    shininess=128,
    custom_filename=None,
    composite_background=True,
    background_path=None,
    mask_path=None,
):
    os.makedirs(output_dir, exist_ok=True)

    # --- Load inputs ---
    normal_bgr = cv2.imread(normal_map_path)
    normal_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    albedo = cv2.imread(albedo_map_path).astype(np.float32) / 255.0

    # --- Normalize normals ---
    normals = normal_rgb * 2.0 - 1.0
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-8)

    H, W, _ = normals.shape
    n_frames = duration * fps

    # Optional background & mask loading
    if composite_background:
        if background_path is None or mask_path is None:
            raise ValueError("To composite background, both background_path and mask_path must be provided.")

        background = cv2.imread(background_path).astype(np.float32) / 255.0
        background = cv2.resize(background, (W, H))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)[..., np.newaxis]
    else:
        mask = np.ones((H, W, 1), dtype=np.float32)  # full face only

    # --- Light directions for 360° horizontal orbit ---
    angles = np.linspace(np.pi, -np.pi, n_frames)
    light_dirs = [np.array([np.cos(a), 0, np.sin(a)], dtype=np.float32) for a in angles]
    view_dir = np.array([0, 0, 1], dtype=np.float32)

    # --- Spotlight vignette ---
    Y, X = np.ogrid[:H, :W]
    center_y, center_x = H / 2, W / 2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    spotlight_mask = 1 - np.clip(dist / (0.8 * max(H, W)), 0, 1)
    spotlight_mask = spotlight_mask[..., np.newaxis]
    spotlight_mask = (1.0 - top_light_strength) + top_light_strength * spotlight_mask

    # --- Output video writer ---
    filename = custom_filename if custom_filename else "phong_metallic.mp4"
    if not filename.lower().endswith(".mp4"):
        filename += ".mp4"

    out_path = os.path.join(output_dir, filename)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"\n🎥 Rendering: Phong Metallic {'with' if composite_background else 'without'} Background")
    for L in tqdm(light_dirs, desc=f"  → {filename}"):
        shaded = phong_metal_shading(normals, L, view_dir, albedo, shininess=shininess)
        shaded *= spotlight_mask

        # Suppress artifacts in dark albedo zones
        dark_mask = (np.sum(albedo, axis=2) < 0.05)[..., np.newaxis]
        shaded[dark_mask[..., 0]] = [0.0, 0.0, 0.0]

        # Composite or not
        if composite_background:
            composite = shaded * mask + background * (1 - mask)
        else:
            composite = shaded * mask  # masked face only on black

        frame = (composite * 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"✅ Saved: {out_path}")
