import cv2
import numpy as np
import os
from tqdm import tqdm
from plyfile import PlyData, PlyElement

# --- Config ---
normal_map_path = "output/photometric_stereo_results/normal_map.png"
albedo_map_path = "output/photometric_stereo_results/albedo_map.png"
output_dir = "output/depth_reconstruction"
os.makedirs(output_dir, exist_ok=True)


def load_normal_map(path):
    """Load and normalize the RGB normal map into unit surface normals in [-1, 1]."""
    normal_bgr = cv2.imread(path).astype(np.float32) / 255.0
    normal_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
    normals = normal_rgb * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    return normals / (norm + 1e-8)


def compute_gradients(normals):
    """Compute surface gradients (p = ∂z/∂x, q = ∂z/∂y) from normals."""
    Nx, Ny, Nz = normals[..., 0], normals[..., 1], normals[..., 2]
    Nz[Nz == 0] = 1e-6  # Prevent division by zero
    p = -Nx / Nz
    q = -Ny / Nz
    return p, q


def frankot_chellappa(p, q):
    """Integrate surface gradients using the Frankot-Chellappa algorithm (Fourier domain)."""
    H, W = p.shape
    fx = np.fft.fftfreq(W)
    fy = np.fft.fftfreq(H)
    u, v = np.meshgrid(fx, fy)
    u = np.fft.fftshift(u)
    v = np.fft.fftshift(v)

    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)

    denom = (2j * np.pi * u)**2 + (2j * np.pi * v)**2
    denom[denom == 0] = 1e-6

    Z = (-2j * np.pi * u * P - 2j * np.pi * v * Q) / denom
    z = np.real(np.fft.ifft2(Z))
    return z


def save_depth_map(depth, path):
    """Normalize and save the depth map as an image."""
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(path, depth_img)


def export_point_cloud(depth_map, albedo_map, output_path, scale=1.0):
    """Convert depth map + albedo map to a colored PLY point cloud."""
    H, W = depth_map.shape
    points = []

    print(f"\n💾 Generating colored point cloud ({H * W} points)...")
    for y in tqdm(range(H), desc="  → Converting to colored point cloud"):
        for x in range(W):
            z = depth_map[y, x] * scale
            b, g, r = albedo_map[y, x]
            points.append((x, y, z, r, g, b))

    vertex = np.array(points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
    ply.write(output_path)


# --- Main Pipeline ---
print("\n Reconstructing depth from normal map...")
normals = load_normal_map(normal_map_path)
p, q = compute_gradients(normals)
depth_map = frankot_chellappa(p, q)

# --- Load Albedo ---
albedo_bgr = cv2.imread(albedo_map_path)
albedo_rgb = cv2.cvtColor(albedo_bgr, cv2.COLOR_BGR2RGB)

# --- Save Outputs ---
depth_path = os.path.join(output_dir, "depth_map.png")
ply_path = os.path.join(output_dir, "point_cloud_colored.ply")

save_depth_map(depth_map, depth_path)
export_point_cloud(depth_map, albedo_rgb, ply_path, scale=0.5)

print(" Depth map saved to:", depth_path)
print(" Colored point cloud saved to:", ply_path)
