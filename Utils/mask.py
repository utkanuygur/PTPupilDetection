import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image

def create_masks(directory):
    masks_dir = os.path.join(directory, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    images_dir = os.path.join(directory, 'images')
    for image_filename in os.listdir(images_dir):
        if image_filename.endswith('.jpg'):
            image_path = os.path.join(images_dir, image_filename)
            csv_filename = image_filename.replace('.jpg', '.csv')
            csv_path = os.path.join(directory, 'labels', csv_filename)
            
            if not os.path.exists(csv_path):
                print(f"CSV file {csv_filename} not found, skipping.")
                continue
            
            labels_df = pd.read_csv(csv_path)
            row = labels_df.iloc[0]
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            mask = np.zeros_like(image)
            
            center = (int(row['Center_X']), int(row['Center_Y']))
            axes = (int(row['Major_Axis'] / 2), int(row['Minor_Axis'] / 2))
            angle = row['Angle']
            
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
            
            mask_pil = Image.fromarray(mask)
            
            mask_filename = os.path.splitext(image_filename)[0] + '.gif'
            mask_path = os.path.join(masks_dir, mask_filename)
            mask_pil.save(mask_path)
            
    print("Masks created successfully.")

directory = 'confirmed_data/val'
create_masks(directory)
