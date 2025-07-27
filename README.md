# Photometric-Stereo-Relighting
This codebase is part of a project for the **"Computational Photography"** course at the **University of Bonn**, conducted during the **Summer 2025 semester** under the supervision of **[Prof. Dr. Matthias B. Hullin](https://light.informatik.uni-bonn.de/team/matthias-hullin/)**. 

---

The project explores a full pipeline for **relighting objects** using **photometric stereo** under spherical gradient illumination. It focuses on:

- Reconstructing **surface normals** and **albedo maps** from multi-light image sets  
- Rendering realistic and **stylized relighting** animations (Lambertian, quantized, metallic)  
- Performing **3D depth reconstruction** via normal integration (Frankot-Chellappa method)  
- Compositing relit objects into synthetic **backgrounds with masking support**

This repository is organized for modular experimentation — from raw image processing to animated relighting and 3D visualization.

---

## 🎯 Key Features

- 📸 **Photometric Stereo**: Extracts per-pixel normals and albedo from 6-directional gradient-lit 16-bit TIFFs  
- 🌈 **Relighting**: Supports grayscale, RGB, stylized, and metallic relighting  
- 🎞 **Animations**: Generates smooth rotating light animations under different material models  
- 🧠 **Depth Recovery**: Converts normals to 3D depth maps using frequency-domain integration  
- 🖼 **Compositing**: Masks and blends relit subjects into custom backgrounds  
- 🛠 **Tools**: Albedo generation, MP4-to-GIF conversion, GIF frame preview and export  

---

## 📁 Folder Overview

```text
Root Directory 
├── raw_input/                  	# Original input images (ARW or JPG)
│   ├── linear-gradient-patterns/
│   └── rgb-gradient-patterns/  	# (Unused)
│
├── processed/                  	# Intermediate processing results
│   ├── preview_8bit/           	# Viewable preview of processed TIFFs
│   ├── processed_tiffs/        	# RAWs converted to linear TIFFs
│   ├── roi_cropped_tiffs/      	# Cropped & masked TIFFs based on ROI
│   ├── masks/
│   └── background image/
│
├── output/                     	# Final project outputs
    ├── animations/             	# MP4 relighting animations
    ├── depth_reconstruction/   	# Depth map & point cloud
    ├── generated_albedos/      	# Grayscale, RGB & custom albedos
    ├── lambertian_relighting_images/
    └── photometric_stereo_results/  	# Normals + albedo maps
```
---

## 🧪 Required Libraries

Install all dependencies using:

```bash
pip install numpy opencv-python matplotlib pillow tqdm moviepy plyfile
```

---

## Codebase Structure & Instructions

### Main Files:

```text
- preprocessing.py 				# If processing of RAW data is necessary then start with data processing part.
- photometric_stereo.py 			# Normal and Albedo maps are generated with processed data (in his case cropped 16 bit tiff files).
- generate_relight_images.py 			# For relight rendered images both albedo and normal maps are necessary (includes rgb/gray options).
- animate.py 					# In case normal, albedo (optional: background, mask) available, generating animation can be done from here.
- animate_standard_diffused_lambertian.py 
- animate_stylized_quantized_diffuse.py
- animate_phong_metalic.py

- reconstruct_depth_from_normals.py 		# In case necessary it generates depth map. 
```

## Helper Files:

```text
- generate_albedos.py 				# Creates custom albedo (currently generates 3 types). 
- mp4_to_gif.py 				# To convert mp4 to gif format.
- extract_gif_frames.py 			# To extract gif frames, preview and save.
```

## For Debugging and Analysis:

```text
- some_check_ups.py
- compare_albedos.py
```



