import cv2
import pandas as pd
import os

def draw_ellipse(img_path, csv_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        return

    try:
        csv_data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {csv_path}")
        print(e)
        return

    try:
        cx = int(csv_data["Center_X"][0])
        cy = int(csv_data["Center_Y"][0])
        major = int(csv_data["Major_Axis"][0]) // 2
        minor = int(csv_data["Minor_Axis"][0]) // 2
        angle = float(csv_data["Angle"][0])
    except KeyError as e:
        print(f"Missing expected column in CSV file: {csv_path} for the parameter: {e}")
        return

    cv2.ellipse(img, (cx, cy), (major, minor), angle, 0, 360, (0, 255, 0), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

def process(imgs_dir, lbls_dir, out_dir):
    for root, _, files in os.walk(imgs_dir):
        for img_file in files:
            if img_file.endswith(".jpg"):
                img_path = os.path.join(root, img_file)
                relative_path = os.path.relpath(img_path, imgs_dir)
                csv_file = os.path.splitext(relative_path)[0] + ".csv"
                csv_path = os.path.join(lbls_dir, csv_file)
                out_path = os.path.join(out_dir, relative_path)
                draw_ellipse(img_path, csv_path, out_path)

def main(base_dir):
    os.makedirs(os.path.join(base_dir, "output"), exist_ok=True)
    imgs_dir = os.path.join(base_dir, "images")
    lbls_dir = os.path.join(base_dir, "labels")
    out_dir = os.path.join(base_dir, "output")
    process(imgs_dir, lbls_dir, out_dir)

if __name__ == "__main__":
    main("confirmed_data/val")
