import os
from tkinter import Tk, filedialog
from PIL import Image
import matplotlib.pyplot as plt

def select_gif_file():
    root = Tk()
    root.withdraw()
    gif_path = filedialog.askopenfilename(
        title="Select a GIF file",
        filetypes=[("GIF files", "*.gif")],
        initialdir="output/animations"
    )
    return gif_path

def display_frame(img, frame_idx):
    img.seek(frame_idx)
    plt.imshow(img.convert("RGB"))
    plt.title(f"Frame {frame_idx + 1}")  # Display as 1-based
    plt.axis("off")
    plt.show()

def save_frame(img, frame_idx, output_dir):
    img.seek(frame_idx)
    img.convert("RGB").save(os.path.join(output_dir, f"frame_{frame_idx + 1}.png"))
    print(f"✅ Saved: frame_{frame_idx + 1}.png")

def main():
    gif_path = select_gif_file()
    if not gif_path:
        print("No GIF selected. Exiting.")
        return

    img = Image.open(gif_path)
    total_frames = img.n_frames
    print(f"📽️ Total frames in GIF: {total_frames}")

    while True:
        frame_to_preview = input("Enter frame number to preview (1-based, or 'done' to stop): ").strip()
        if frame_to_preview.lower() == "done":
            break
        if frame_to_preview.isdigit():
            idx = int(frame_to_preview) - 1
            if 0 <= idx < total_frames:
                display_frame(img, idx)
            else:
                print(f"Frame number out of range. Valid range: 1 to {total_frames}")
        else:
            print("Please enter a valid number or 'done'.")

    selected_frames = input("Enter frame numbers to save (comma-separated, 1-based): ").strip()
    if not selected_frames:
        print("No frames selected. Exiting.")
        return

    raw_indices = [i.strip() for i in selected_frames.split(",") if i.strip().isdigit()]
    frame_indices = [int(i) - 1 for i in raw_indices]
    valid_indices = [i for i in frame_indices if 0 <= i < total_frames]
    invalid_indices = [int(i) + 1 for i in frame_indices if i not in valid_indices]

    output_dir = os.path.join(os.path.dirname(gif_path), "extracted_frames")
    os.makedirs(output_dir, exist_ok=True)

    for idx in valid_indices:
        save_frame(img, idx, output_dir)

    if invalid_indices:
        print(f"Skipped invalid frame numbers: {invalid_indices}")

    print(f"\nAll saved frames are in: {output_dir}")

if __name__ == "__main__":
    main()
