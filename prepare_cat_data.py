import os
import shutil
import random
from glob import glob

# Gather all jpg images from all CAT_XX folders
cat_folders = [os.path.join('archive', 'cats', f'CAT_0{i}') for i in range(7)]
all_images = []
for folder in cat_folders:
    all_images.extend(glob(os.path.join(folder, '*.jpg')))

print(f"Found {len(all_images)} images.")

# Shuffle and split
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

# Prepare output dirs
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)

# Copy images
for img in train_imgs:
    shutil.copy(img, os.path.join('data/train', os.path.basename(img)))
for img in val_imgs:
    shutil.copy(img, os.path.join('data/val', os.path.basename(img)))

print(f"Copied {len(train_imgs)} images to data/train and {len(val_imgs)} images to data/val.") 