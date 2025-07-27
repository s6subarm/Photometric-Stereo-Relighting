# Photometric-Stereo-Relighting
This codebase is part of a project for the **"Computational Photography"** course at the **University of Bonn**, conducted during the **Summer 2025 semester** under the supervision of **[Prof. Dr. Matthias B. Hullin](https://light.informatik.uni-bonn.de/team/matthias-hullin/)**. The project focuses on photometric stereo reconstruction, object relighting, and 3D depth recovery from gradient-illuminated images.

_____________________________________________________________________________________________________

# Directory Structure 
_____________________________________________________________________________________________________

Root Directory 
	|----------> raw_input (input images ARW/JPG)
	|		|
	|		|----------> linear-gradient-patterns 
	|		|----------> rgb-gradient-patterns (unused)
	|
	|
	|----------> processed (raws to tiffs)
	|		|
	|		|----------> preview_8bit (for viewing the processed data)
	|		|----------> processed_tiffs (without masking or crop)
	|		|----------> roi_cropped_tiffs (cropped + masked based on ROI)
	|		|----------> masks 
	|		|----------> background image
	|
	|
	|----------> output (All the results for the Project)
			|
			|----------> animations (mp4 format)
			|----------> depth_reconstruction (depth map + point cloud)
			|----------> generated_albedos (defaults + generated)
			|----------> lambertian_relighting_images (grayscale + rgb)
			|----------> photometric_stereo_results (normal + albedo maps)





_____________________________________________________________________________________________________

# Necessary Libraries for Environment
_____________________________________________________________________________________________________

## Third-Party Library List:
=====================================
- numpy ------------------------------------> All numerical computations across scripts
- opencv-python	----------------------------> All image I/O, processing, and transformations
- matplotlib -------------------------------> Frame previews, visualizations
- Pillow -----------------------------------> Frame-by-frame GIF handling
- tqdm -------------------------------------> Animation progress bars
- moviepy ----------------------------------> MP4 to GIF conversion, grid combining
- plyfile ----------------------------------> Writing 3D PLY point clouds
=====================================




## Installation Command:
=========================================================================================
|											|
|	pip install numpy opencv-python matplotlib pillow tqdm moviepy plyfile		|
|											|
=========================================================================================





_____________________________________________________________________________________________________

# Code Base Structure & Instructions.
_____________________________________________________________________________________________________

## Main Files:
=====================================
### Processing Part 
- preprocessing.py -----------------------------> If processing of RAW data is necessary then start 
						  with data processing part.

- photometric_stereo.py ------------------------> Normal and Albedo maps are generated with processed 
						  data (in his case cropped 16 bit tiff files).

- generate_relight_images.py -------------------> For relight rendered images both albedo and normal 
						  maps are necessary (includes rgb/gray options).


### Creating Animation Part
- animate.py ----------------------------------> In case normal, albedo (optional: background, mask) 
	|					 available, generating animation can be done from 
	|					 here.
	|
	|-- animate_standard_diffused_lambertian.py 
	|-- animate_stylized_quantized_diffuse.py
	|-- animate_phong_metalic.py

### Depth Map Part
- reconstruct_depth_from_normals.py -----------> In case necessary it generates depth map. 
=====================================




## Helper Files:
=====================================
- generate_albedos.py -------------------------> Creates custom albedo (currently generates 3 types). 
- mp4_to_gif.py -------------------------------> To convert mp4 to gif format.
- extract_gif_frames.py -----------------------> To extract gif frames, preview and save.
=====================================




## For Debugging and Analysis:
=====================================
- some_check_ups.py
- compare_albedos.py
=====================================








