from datasets import load_dataset
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

def convert_to_yolo_format(dataset, output_dir, split_name='train'):
    images_dir = Path(output_dir) / split_name / 'images'
    labels_dir = Path(output_dir) / split_name / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(dataset):
        img = sample['image']
        img_width, img_height = img.size
        img_path = images_dir / f"{idx:06d}.jpg"
        img.save(img_path)
        label_path = labels_dir / f"{idx:06d}.txt"
        with open(label_path, 'w') as f:
            for bbox, category in zip(sample['objects']['bbox'], sample['objects']['category']):
                x, y, w, h = bbox
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                f.write(f"{category} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def create_data_yaml(output_dir):
    data_config = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 5,
        'names': {
            0: 'Coverall',
            1: 'Face_Shield',
            2: 'Gloves',
            3: 'Goggles',
            4: 'Mask'
        }
    }
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    print(f"Created data.yaml at: {yaml_path}")

def split_dataset(cppe5_train, output_dir):
    total_samples = len(cppe5_train)
    indices = list(range(total_samples))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.33, random_state=42)
    train_data = [cppe5_train[i] for i in train_idx]
    val_data = [cppe5_train[i] for i in val_idx]
    test_data = [cppe5_train[i] for i in test_idx]
    return train_data, val_data, test_data

def prepare_dataset():
    output_dir = 'data'
    cppe5 = load_dataset("cppe-5")
    train_data, val_data, test_data = split_dataset(cppe5['train'], output_dir)
    convert_to_yolo_format(train_data, output_dir, 'train')
    convert_to_yolo_format(val_data, output_dir, 'val')
    convert_to_yolo_format(test_data, output_dir, 'test')
    create_data_yaml(output_dir)

if __name__ == "__main__":
    prepare_dataset()
