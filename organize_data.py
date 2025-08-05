import os
import shutil
from sklearn.model_selection import train_test_split
import random

def organize_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15):
    os.makedirs(dest_dir, exist_ok=True)
    classes = ['Cat', 'Dog']  # Match Kaggle dataset folder names
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls.lower()), exist_ok=True)
    
    for cls in classes:
        # Get all images in the class folder
        class_path = os.path.join(source_dir, 'PetImages', cls)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.png'))]
        
        # Filter out corrupted or invalid images
        valid_images = []
        for img in images:
            try:
                with open(img, 'rb') as f:
                    f.read(1024)  # Attempt to read file
                valid_images.append(img)
            except:
                print(f"Skipping corrupted file: {img}")
        
        # Split images
        train_imgs, temp_imgs = train_test_split(valid_images, train_size=train_ratio, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=val_ratio/(1-train_ratio), random_state=42)
        
        # Copy images to respective folders
        for split, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            for img in img_list:
                dest_path = os.path.join(dest_dir, split, cls.lower(), os.path.basename(img))
                shutil.copy(img, dest_path)
                print(f"Copied {img} to {dest_path}")

if __name__ == "__main__":
    source_dir = "raw_data"  # Path where Kaggle dataset is extracted
    dest_dir = "data"        # Destination for organized data
    organize_dataset(source_dir, dest_dir)