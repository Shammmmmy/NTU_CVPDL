import os
import random
import shutil

img_dir = "./datasets/images/train/img" 
label_dir = "./datasets/labels/train/img"
dataset_dir = "./datasets"

# val split ratio
split_ratio = 0.2 

# make target directories
target_dirs = {
    "train_images": os.path.join(dataset_dir, "images", "train", "img"),
    "val_images": os.path.join(dataset_dir, "images", "val", "img"),
    "train_labels": os.path.join(dataset_dir, "labels", "train", "img"),
    "val_labels": os.path.join(dataset_dir, "labels", "val", "img"),
}

for dir_path in target_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Get a list of all image files and shuffle them
image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)

# calculate the number of validation files
num_val_files = int(len(image_files) * split_ratio)
val_files = image_files[:num_val_files]
train_files = image_files[num_val_files:]

print(f"Total number of pictures: {len(image_files)}")
print(f"Number of training pictures: {len(train_files)} ({100 - split_ratio*100}%)")
print(f"Number of validation pictures: {len(val_files)} ({split_ratio*100}%)")

# move files function
def move_files(file_list, dest_img_dir, dest_lbl_dir):
    for file_name in file_list:
        base_name = os.path.splitext(file_name)[0]
        label_name = f"{base_name}.txt"
        
        # source and destination image paths
        source_img_path = os.path.join(img_dir, file_name)
        dest_img_path = os.path.join(dest_img_dir, file_name)
        
        # annotation and destination label paths
        source_lbl_path = os.path.join(label_dir, label_name)
        dest_lbl_path = os.path.join(dest_lbl_dir, label_name)
        
        # check if the image already exists in the destination
        if not os.path.exists(dest_img_path):
            shutil.move(source_img_path, dest_img_path)
            
            # check and move annotation files at the same time
            if os.path.exists(source_lbl_path) and not os.path.exists(dest_lbl_path):
                shutil.move(source_lbl_path, dest_lbl_path)
            elif not os.path.exists(source_lbl_path):
                print(f"Unable to find annotation file for image '{file_name}'.")

if __name__ == '__main__':
    move_files(train_files, target_dirs["train_images"], target_dirs["train_labels"])
    move_files(val_files, target_dirs["val_images"], target_dirs["val_labels"])

    print("data split completed.\n")