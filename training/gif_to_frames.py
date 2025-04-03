import os
from PIL import Image
import glob
import re

# === Paths ===
base_dir = r"C:\Users\abiva\OneDrive\Desktop\FormSenseAI_new"
input_dir = os.path.join(base_dir, "training", "gifs")
output_dir = os.path.join(base_dir, "training", "frames")

os.makedirs(output_dir, exist_ok=True)

# === Clean filename ===
def clean_name(name):
    name = os.path.splitext(name)[0]            # Remove .gif
    name = re.sub(r"[^\w\s]", "", name)         # Remove special characters
    return name.strip()

# === Process GIFs ===
for gif_path in glob.glob(os.path.join(input_dir, "*.gif")):
    original_name = os.path.basename(gif_path)
    base_name = clean_name(original_name)
    save_folder = os.path.join(output_dir, base_name)
    os.makedirs(save_folder, exist_ok=True)

    with Image.open(gif_path) as im:
        frame = 0
        try:
            while True:
                im.seek(frame)
                frame_image = im.convert("RGB")
                filename = f"{base_name} {frame + 1} frame.jpg"
                frame_path = os.path.join(save_folder, filename)
                frame_image.save(frame_path)
                print(f"✅ Saved: {frame_path}")
                frame += 1
        except EOFError:
            print(f"✔️ Done: {frame} frames from '{original_name}'")
