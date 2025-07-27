import os
import math
import tkinter as tk
from tkinter import filedialog
from moviepy import VideoFileClip, clips_array, ColorClip  # MoviePy v2.0+ 

def prompt_float(message, default):
    value = input(f"{message} (default: {default}): ").strip()
    return float(value) if value else default

def prompt_int(message, default):
    value = input(f"{message} (default: {default}): ").strip()
    return int(value) if value else default

def convert_to_gif(mp4_path, output_dir="output/animations", fps=15, scale=0.2):
    if not os.path.exists(mp4_path):
        print(f"File not found: {mp4_path}")
        return

    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    output_path = os.path.join(output_dir, base_name + ".gif")

    print(f"Converting: {mp4_path} → {output_path}")

    try:
        clip = VideoFileClip(mp4_path)
        if scale != 1.0:
            clip = clip.resized(scale)
        clip.write_gif(output_path, fps=fps)
        print(f"✅ Saved GIF: {output_path}")
    except Exception as e:
        print(f"Conversion failed for {mp4_path}: {e}")

def combine_and_convert(mp4_paths, output_path, fps=15, scale=1.0, per_row=2):
    print("Combining videos into grid...")
    clips = []

    for path in mp4_paths:
        clip = VideoFileClip(path)
        if scale != 1.0:
            clip = clip.resized(scale)
        clips.append(clip)

    n = len(clips)
    rows = math.ceil(n / per_row)
    grid = []

    for r in range(rows):
        row_clips = clips[r * per_row:(r + 1) * per_row]
        if len(row_clips) < per_row:
            w, h = row_clips[0].size
            duration = row_clips[0].duration
            blank = ColorClip(size=(w, h), color=(0, 0, 0), duration=duration)
            row_clips += [blank] * (per_row - len(row_clips))
        grid.append(row_clips)

    final_clip = clips_array(grid)
    final_clip.write_gif(output_path, fps=fps)
    print(f"✅ Saved combined GIF: {output_path}")

def manual_file_selection(n, initial_dir="output/animations"):
    paths = []
    for i in range(n):
        print(f"Select {i+1}{get_suffix(i+1)} file.")
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title=f"Select MP4 file #{i+1}",
            filetypes=[("MP4 Video Files", "*.mp4")],
            initialdir=initial_dir
        )
        if not file_path:
            print("No file selected. Aborting.")
            return []
        paths.append(file_path)
    return paths

def get_suffix(n):
    return {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th")

def select_videos_and_convert():
    side_by_side = input("Do you want to create gif files side by side? [y/n]: ").strip().lower() == "y"

    if side_by_side:
        per_row = prompt_int("How many GIFs per row (max 4)", 2)
        per_row = min(max(per_row, 1), 4)

        total = prompt_int("How many GIFs to include in total?", per_row)
        mp4_paths = manual_file_selection(total)

        if per_row > 4:
            print("⚠️ Max allowed is 4 GIFs per row. The rest will be pushed down to next row.")

        if not mp4_paths or len(mp4_paths) != total:
            print("Not all files selected.")
            return

        fps = prompt_int("Frames per second", 30)
        scale = prompt_float("Resize scale (e.g., 0.5 for half-size)", 1.0)

        print("💾 Choose output location and filename for combined GIF")
        root = tk.Tk()
        root.withdraw()
        output_path = filedialog.asksaveasfilename(
            title="Save combined GIF as",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif")],
            initialdir="output/animations"
        )

        if not output_path:
            print("No output path selected. Aborting.")
            return

        combine_and_convert(mp4_paths, output_path=output_path, fps=fps, scale=scale, per_row=per_row)

    else:
        # Select individual files FIRST
        root = tk.Tk()
        root.withdraw()
        files = filedialog.askopenfilenames(
            title="Select MP4 video(s) to convert",
            filetypes=[("MP4 Video Files", "*.mp4")],
            initialdir="output/animations"
        )

        if not files:
            print("No files selected.")
            return

        # Then ask for FPS and scale
        print("GIF Conversion Settings")
        fps = prompt_int("Frames per second", 30)
        scale = prompt_float("Resize scale (e.g., 0.5 for half-size)", 1.0)

        for path in files:
            convert_to_gif(path, output_dir="output/animations", fps=fps, scale=scale)

if __name__ == "__main__":
    select_videos_and_convert()
