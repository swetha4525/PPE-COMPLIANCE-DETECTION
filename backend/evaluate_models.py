from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import time

def evaluate_model(model_path, data_yaml, model_name):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split='test')
    test_images = list(Path('data/test/images').glob('*.jpg'))[:100]
    start_time = time.time()
    for img_path in test_images:
        model(img_path, verbose=False)
    elapsed_time = time.time() - start_time
    fps = len(test_images) / elapsed_time
    model_size = Path(model_path).stat().st_size / (1024 * 1024)
    results_dict = {
        'Model': model_name,
        'mAP@0.5': metrics.box.map50,
        'mAP@0.5:0.95': metrics.box.map,
        'Precision': metrics.box.mp,
        'Recall': metrics.box.mr,
        'FPS': fps,
        'Model Size (MB)': model_size
    }
    return results_dict

def compare_models():
    models = {
        'YOLOv8n': 'runs/train/yolov8n_cppe5/weights/best.pt',
        'YOLOv5s': 'runs/train/yolov5s_cppe5/weights/best.pt'
    }
    data_yaml = 'data/data.yaml'
    results = []
    for model_name, model_path in models.items():
        if Path(model_path).exists():
            result = evaluate_model(model_path, data_yaml, model_name)
            results.append(result)
    df = pd.DataFrame(results)
    df.to_csv('model_comparison_results.csv', index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    compare_models()
