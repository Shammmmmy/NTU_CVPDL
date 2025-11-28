import os
import torch 
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

# model prediction
def model_predict(image_path, model_path="runs/detect/train/weights/best.pt"):
    model = YOLO(model_path)
    submission_data = []
    results = model.predict(
        source=image_path,  
        conf=0.001, 
        iou=0.7, # best
        device=device,
        imgsz=1920,
        max_det=500,
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
                cls = box.cls.cpu().item()
                
                x_c_norm, y_c_norm, w_norm, h_norm = xywhn

                w_pix = w_norm * W
                h_pix = h_norm * H
                
                bb_left = (x_c_norm - w_norm / 2) * W
                bb_top = (y_c_norm - h_norm / 2) * H

                conf_str = f"{conf:.{6}f}"
                bb_left_str = f"{bb_left:.{2}f}"
                bb_top_str = f"{bb_top:.{2}f}"
                w_pix_str = f"{w_pix:.{2}f}"
                h_pix_str = f"{h_pix:.{2}f}"
               
                single_pred_string = (
                    f"{conf_str} {bb_left_str} {bb_top_str} "
                    f"{w_pix_str} {h_pix_str} {int(cls)}"
                )
                all_predictions.append(single_pred_string)

            prediction_string = " ".join(all_predictions)
        else:
            prediction_string = ""
        
        submission_data.append([image_id, prediction_string])
        image_index += 1
        print(f"Processing image {image_id}: Detected {len(result.boxes)} objects")
    
    submission_df = pd.DataFrame(submission_data, columns=["Image_ID", "PredictionString"])
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' has been created.")

if __name__ == "__main__":
    device = "0" if torch.cuda.is_available() else "cpu"
    image_path = "./datasets/images/test"
    model_predict(image_path, model_path="runs/detect/train/weights/best.pt")