import os
import random
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Data preprocessing function
def preprocess_data(gt_path, img_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(gt_path) if f.endswith('.txt')]
    sorted_txt_files = sorted(txt_files)
    
    for txt_file in tqdm(sorted_txt_files, desc="Processing annotations"):
        df = pd.read_csv(os.path.join(gt_path, txt_file), header=None, names=['class_id', 'left', 'top', 'width', 'height'])
        df = df[(df['width'] > 0) & (df['height'] > 0)]

        img_name = txt_file.replace('.txt', '.png')
        img_path = os.path.join(img_dir, img_name)
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        yolo_lines = []
        for _, row in df.iterrows():
            cx = (row['left'] + row['width'] / 2) / img_w
            cy = (row['top'] + row['height'] / 2) / img_h
            w = row['width'] / img_w
            h = row['height'] / img_h
            cx, cy, w, h = [max(0, min(1, v)) for v in [cx, cy, w, h]]
            yolo_lines.append(f"{int(row['class_id'])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path = os.path.join(label_dir, txt_file)
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        os.remove(os.path.join(gt_path, txt_file))
    print(f"YOLO format annotations have been saved to: {label_dir}")

# move files function
def move_files(file_list, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
    for file_name in file_list:
        base_name = os.path.splitext(file_name)[0]
        label_name = f"{base_name}.txt"
        
        # source and destination image and annotation and destination label paths
        source_img_path = os.path.join(source_img_dir, file_name)
        dest_img_path = os.path.join(dest_img_dir, file_name)
        source_lbl_path = os.path.join(source_lbl_dir, label_name)
        dest_lbl_path = os.path.join(dest_lbl_dir, label_name)
        
        # check if the image already exists in the destination
        if not os.path.exists(dest_img_path):
            shutil.move(source_img_path, dest_img_path)
            if os.path.exists(source_lbl_path) and not os.path.exists(dest_lbl_path):
                shutil.move(source_lbl_path, dest_lbl_path)
            elif not os.path.exists(source_lbl_path):
                print(f"Unable to find annotation file for image '{file_name}'.")

# Data splitting function
def split_data(img_dir, label_dir, dataset_dir, split_ratio):
    target_dirs = {
        "train_images": os.path.join(dataset_dir, "images", "train"),
        "val_images": os.path.join(dataset_dir, "images", "val"),
        "train_labels": os.path.join(dataset_dir, "labels", "train"),
        "val_labels": os.path.join(dataset_dir, "labels", "val"),
    }
    for dir_path in target_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    class_to_images = {0: [], 1: [], 2: [], 3: []}  # car, hov, person, motorcycle
    
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                classes_in_image = set()
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        classes_in_image.add(class_id)

                if classes_in_image:
                    primary_class = min(classes_in_image)
                    class_to_images[primary_class].append(img_file)
                else:
                    print(f"Warning: No valid classes found in {label_file}")
                    f.close()
                    os.remove(os.path.join(img_dir, img_file))
                    os.remove(label_path)

    train_files = []
    val_files = []

    class_names = ["car", "hov", "person", "motorcycle"]
    for class_id, images in class_to_images.items():
        random.shuffle(images)
        num_val = max(1, int(round(len(images) * split_ratio))) if len(images) >= 2 else 0
        val_files.extend(images[:num_val])
        train_files.extend(images[num_val:])
        print(f"Class {class_id} ({class_names[class_id]}): Total={len(images)}, Train={len(images)-num_val}, Val={num_val}")

    print(f"\nNumber of training pictures: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"Number of validation pictures: {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)")

    move_files(train_files, img_dir, label_dir, target_dirs["train_images"], target_dirs["train_labels"])
    move_files(val_files, img_dir, label_dir, target_dirs["val_images"], target_dirs["val_labels"])
    print("Data split completed.\n")
    

# Create data.yaml
def create_yaml(save_path="./data.yaml", class_name=["car", "hov", "person", "motorcycle"]):
    content = f"""path: ../datasets
train: images/train
val: images/val
test: images/test

nc: {len(class_name)}
names: {class_name}"""

    with open(save_path, 'w') as f:
        f.write(content)
    print(f"data.yaml has been created at: {save_path}")

if __name__ == "__main__":
    img_dir = './datasets/images/train'
    gt_path = './datasets/images/train'
    label_dir = './datasets/labels/train'
    dataset_dir = "./datasets"

    preprocess_data(gt_path, img_dir, label_dir)
    split_data(img_dir, label_dir, dataset_dir, split_ratio=0.1)
    create_yaml(save_path="./data.yaml", class_name=["car", "hov", "person", "motorcycle"])
 