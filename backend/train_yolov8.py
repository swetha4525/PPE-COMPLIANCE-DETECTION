from ultralytics import YOLO
import torch

def train_yolov8():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='data/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.01,
        optimizer='SGD',
        patience=50,
        device=device,
        project='runs/train',
        name='yolov8n_cppe5',
        exist_ok=True
    )
    metrics = model.val()
    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)
    model.export(format='onnx')
    return model, metrics

if __name__ == "__main__":
    train_yolov8()
