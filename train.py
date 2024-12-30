import os
import yaml
from ultralytics import YOLO
from termcolor import colored

def train_model():
    """Train the YOLO model on collected data"""
    # Create dataset directories if they don't exist
    data_dir = "lib/data"
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(colored("[ERROR] No training data found. Run the aimbot in collect_data mode first!", "red"))
        print(colored("Usage: python lunar.py collect_data", "yellow"))
        return
        
    image_count = len(os.listdir(images_dir))
    if image_count == 0:
        print(colored("[ERROR] No training images found in lib/data/images/", "red"))
        return
        
    print(colored(f"[INFO] Found {image_count} training images", "green"))
    
    # Create dataset.yaml
    dataset_config = {
        'path': 'lib/data',
        'train': 'images',
        'val': 'images',
        'names': ['player']
    }
    
    with open('lib/data/dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    print(colored("[INFO] Starting LunarPLUS model training...", "green"))
    
    # Initialize and train YOLO model
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model
    try:
        model.train(
            data='lib/data/dataset.yaml',
            epochs=50,
            imgsz=640,
            batch=16,
            name='lunarplus_custom'
        )
        
        # Move the best model to replace the current one
        best_model = 'runs/detect/lunarplus_custom/weights/best.pt'
        if os.path.exists(best_model):
            os.replace(best_model, 'lib/best.pt')
            print(colored("[INFO] Training complete! New LunarPLUS model saved as lib/best.pt", "green"))
        else:
            print(colored("[ERROR] Training failed or no best model found", "red"))
            
    except Exception as e:
        print(colored(f"[ERROR] Training failed: {str(e)}", "red"))

if __name__ == "__main__":
    train_model()
