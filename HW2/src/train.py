import os
import torch 
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

# model training
def model_training(device):
    model = YOLO("yolov8x.yaml")

    train_results = model.train(
        data="./data.yaml",  
        epochs=400,  
        imgsz=1920,  
        device=device, 
        batch=1,
        patience=30,
        pretrained=False,
        verbose=True,

        # augmentations
        mosaic=1.0,
        mixup=0.3,
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
def model_evaluate(best_model_path):
    model = YOLO(best_model_path)
    metrics = model.val()
    mAP_50_95 = metrics.box.map  # mAP for 0.50:0.95 IoU 
    print(f"The model's mAP@0.50:0.95 on the validation set is: {mAP_50_95:.4f}")

if __name__ == "__main__":
    device = "0" if torch.cuda.is_available() else "cpu"
    training_results = model_training(device)
    results_save_path = training_results.save_dir
    best_model_path = os.path.join(results_save_path, "weights", "best.pt")
    plot_training_results(results_save_path)
    model_evaluate(best_model_path)