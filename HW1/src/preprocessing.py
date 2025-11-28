import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

def preprocess_data(gt_path, img_dir, output_dir, class_id=0):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(gt_path, header=None, names=['image_id', 'left', 'top', 'width', 'height'])
    print(f"Original number of annotations: {len(df)}")
    print(df.head())

    # Remove data where the BoundingBox width or height equals 0
    df = df[(df['width'] > 0) & (df['height'] > 0)]
    print(f"Number of annotations after removal: {len(df)}")

    # Remove groundtruths without corresponding images
    print("Check if image files exist")
    unique_images = df['image_id'].unique()
    exist_images = {
        img_id for img_id in tqdm(unique_images, desc="Check images")
        if os.path.exists(os.path.join(img_dir, f"{int(img_id):08d}.jpg"))
    }
    df = df[df['image_id'].isin(exist_images)]
    print(f"Final number of annotations: {len(df)}")

    # Convert to YOLO format
    for img_id, group in tqdm(df.groupby('image_id'), desc="Convert to YOLO"):
        img_path = os.path.join(img_dir, f"{int(img_id):08d}.jpg")
        
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines = []
        for _, row in group.iterrows():
            cx = (row['left'] + row['width'] / 2) / img_w
            cy = (row['top'] + row['height'] / 2) / img_h
            w = row['width'] / img_w
            h = row['height'] / img_h
            cx, cy, w, h = [max(0, min(1, v)) for v in [cx, cy, w, h]]
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Save txt annotation
        label_path = os.path.join(output_dir, f"{int(img_id):08d}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    print(f"YOLO format annotations have been saved to: {output_dir}")

    return df

# Create data.yaml
def create_yaml(save_path="./data.yaml", class_name="pig"):
    content = f"""path: ../datasets
train: images/train/img
val: images/val/img
test: images/test/img

nc: 1
names:
  0: {class_name}
"""
    with open(save_path, "w") as f:
        f.write(content)
    print(f"data.yaml has been created: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    gt_path = './datasets/images/train/gt.txt'
    img_dir = './datasets/images/train/img'
    output_dir = './datasets/labels/train/img'

    # dataset preprocessing
    cleaned_gt = preprocess_data(gt_path, img_dir, output_dir)

    # Create data.yaml
    create_yaml(save_path="./data.yaml", class_name="pig")

