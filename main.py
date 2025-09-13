import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from src.EXR_loader import loadEXR
from src.utils import save_img

import numpy as np
import cv2
import argparse
import re

def EXRToPNG(filepath: str, imgPath: str):
    exrImg = loadEXR(filepath)
    exrImg = exrImg * 255.0
    exrImg = exrImg.astype(np.uint8)  # Fixed: assign the conversion

    save_img(imgPath, exrImg)

def EXRToPNGFolder(input_folder: str, output_folder: str, name: str = None):
    os.makedirs(output_folder, exist_ok=True)
    pattern = None
    if name:
        # Match files like a_b.exr, where a=name and b is an integer
        pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.exr$", re.IGNORECASE)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".exr"):
            if pattern and not pattern.match(filename):
                continue
            exr_path = os.path.join(input_folder, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_folder, png_filename)
            exrImg = loadEXR(exr_path)
            exrImg = exrImg * 255.0
            exrImg = exrImg.astype(np.uint8)
            save_img(png_path, exrImg)
            print(f"Converted: {exr_path} -> {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EXR files in a folder to PNGs")
    parser.add_argument("--input_folder", type=str, help="Folder containing EXR files")
    parser.add_argument("--output_folder", type=str, help="Folder to save PNG files")
    parser.add_argument("--file_type", type=str, default=None, help="String name prefix for files (e.g., 'a' for a_b.exr)")
    args = parser.parse_args()

    EXRToPNGFolder(args.input_folder, args.output_folder, args.file_type)
