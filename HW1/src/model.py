import os
import cv2
import torch 
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt


device = "0" if torch.cuda.is_available() else "cpu"

# model training
def model_training():
    model = YOLO("yolo11x.yaml")

    train_results = model.train(
        data="./data.yaml",  
        epochs=200,  
        imgsz=640,  
        device=device, 
        batch=16,
        patience=30,
        pretrained=False,

        # augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    return train_results

# Plot training results
def plot_training_results(results_path):
    csv_path = os.path.join(results_path, "results.csv")
    results_df = pd.read_csv(csv_path)
    
    results_df.columns = results_df.columns.str.strip()

    epochs = results_df['epoch']

    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    # Box Loss
    axs[0, 0].plot(epochs, results_df['train/box_loss'], label='Train Box Loss')
    axs[0, 0].plot(epochs, results_df['val/box_loss'], label='Validation Box Loss')
    axs[0, 0].set_title('Box Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Class Loss
    axs[0, 1].plot(epochs, results_df['train/cls_loss'], label='Train Class Loss')
    axs[0, 1].plot(epochs, results_df['val/cls_loss'], label='Validation Class Loss')
    axs[0, 1].set_title('Class Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # DFL Loss
    axs[1, 0].plot(epochs, results_df['train/dfl_loss'], label='Train DFL Loss')
    axs[1, 0].plot(epochs, results_df['val/dfl_loss'], label='Validation DFL Loss')
    axs[1, 0].set_title('DFL Loss')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # mAP curves
    axs[1, 1].plot(epochs, results_df['metrics/mAP50-95(B)'], label='mAP@0.50:0.95')
    axs[1, 1].plot(epochs, results_df['metrics/mAP50(B)'], label='mAP@0.50')
    axs[1, 1].set_title('Mean Average Precision (mAP)')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('mAP')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # save the figure
    save_path = os.path.join(results_path, "loss_plot.png")
    plt.savefig(save_path)
    print(f"Training loss plot has been saved to: {save_path}")
    plt.close()

# model evaluation
def model_evaluate():
    model = YOLO("runs/detect/train/weights/best.pt")
    metrics = model.val()

    mAP_50_95 = metrics.box.map  # mAP for 0.50:0.95 IoU
    
    print(f"The model's mAP@0.50:0.95 on the validation set is: {mAP_50_95:.4f}")

# model prediction
def model_predict(image_path, model_path="runs/detect/train/weights/best.pt"):
    model = YOLO(model_path)
    submission_data = []
    results = model.predict(
        source=image_path,  
        conf=0.5, 
        iou=0.5,
        device=device,
        imgsz=640,
        augment=True,
        save=True,  
        verbose=False
    )
    image_index = 1 

    # prediction results processing
    for result in tqdm(results, desc="Processing prediction results"):
        image_id = image_index
        H, W = result.orig_shape
        
        all_predictions = []
        
        # check if there are any detected boxes
        if result.boxes and len(result.boxes.xywhn) > 0:
            for box in result.boxes:
                # Get YOLO Normalized format coordinates (xywhn) and confidence score (conf)
                xywhn = box.xywhn.cpu().tolist()[0]
                conf = box.conf.cpu().item()
                
                x_c_norm, y_c_norm, w_norm, h_norm = xywhn

                w_pix = w_norm * W
                h_pix = h_norm * H
                
                bb_left = (x_c_norm - w_norm / 2) * W
                bb_top = (y_c_norm - h_norm / 2) * H

                # Format confidence score
                conf_str = f"{conf:.{6}f}"
                
                # Format bounding box coordinates and dimensions
                bb_left_str = f"{bb_left:.{2}f}"
                bb_top_str = f"{bb_top:.{2}f}"
                w_pix_str = f"{w_pix:.{2}f}"
                h_pix_str = f"{h_pix:.{2}f}"
               
                # Prediction format: <conf> <bb_left> <bb_top> <bb_width> <bb_height> <class>
                single_pred_string = (
                    f"{conf_str} {bb_left_str} {bb_top_str} "
                    f"{w_pix_str} {h_pix_str} {0}"
                )
                all_predictions.append(single_pred_string)

            # Use spaces to concatenate the properties of all objects.
            prediction_string = " ".join(all_predictions)
        else:
            # If no objects are detected, leave it empty
            prediction_string = ""
        
        # Add the result to the list, Image_ID is the current continuous index
        submission_data.append([image_id, prediction_string])
        
        image_index += 1

    print(f"Processing image {image_id}: Detected {len(result.boxes)} objects")
    submission_df = pd.DataFrame(submission_data, columns=["Image_ID", "PredictionString"])
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' has been created.")

if __name__ == '__main__':
    image_path = "./datasets/images/test/img"
    
    training_results = model_training()
    results_save_path = training_results.save_dir
    plot_training_results(results_save_path)
    model_evaluate()
    model_predict(image_path, model_path="runs/detect/train/weights/best.pt")
    