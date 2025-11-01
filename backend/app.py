import os
import cv2
import numpy as np
from ultralytics import YOLO
import os
import cv2
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATHS = {
    'yolov8': '../models/yolov8.pt',
#    'yolov5': '../models/yolov5.pt',
    'faster_rcnn': '../models/faster_rcnn.pth'
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_yolov8 = YOLO(MODEL_PATHS['yolov8'])
#model_yolov5 = YOLO(MODEL_PATHS['yolov5'])

faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
faster_rcnn_model.load_state_dict(torch.load(MODEL_PATHS['faster_rcnn'], map_location='cpu'))
faster_rcnn_model.eval()

PPE_CLASSES = {
    0: 'Coverall',
    1: 'Face_Shield',
    2: 'Gloves',
    3: 'Goggles',
    4: 'Mask'
}
CONFIDENCE_THRESHOLD = 0.25

def run_faster_rcnn_inference(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255
    image = image.unsqueeze(0)
    with torch.no_grad():
        outputs = faster_rcnn_model(image)[0]
    detections = []
    for idx in range(len(outputs['boxes'])):
        box = outputs['boxes'][idx].detach().cpu().numpy().tolist()
        score = outputs['scores'][idx].item()
        class_id = outputs['labels'][idx].item()
        if score > CONFIDENCE_THRESHOLD:
            detections.append({
                'class_id': class_id,
                'class_name': PPE_CLASSES.get(class_id, f'Class{class_id}'),
                'confidence': score,
                'bbox': box
            })
    return detections

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            import json
            msg = json.loads(data)
            frame_b64 = msg['frame'].split(',')[1]
            model_type = msg['model']
            import base64
            arr = np.frombuffer(base64.b64decode(frame_b64), np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if model_type == 'yolov8':
                results = model_yolov8(frame, conf=0.25, iou=0.45)
                detections = []
                for r in results:
                    for box in r.boxes:
                        detections.append({
                            'class_id': int(box.cls[0]),
                            'class_name': PPE_CLASSES[int(box.cls[0])],
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist()
                        })
     #       elif model_type == 'yolov5':
     #           results = model_yolov5(frame, conf=0.25, iou=0.45)
     #           detections = []
     #           for r in results:
     #               for box in r.boxes:
     #                   detections.append({
     #                       'class_id': int(box.cls[0]),
     #                       'class_name': PPE_CLASSES[int(box.cls[0])],
     #                       'confidence': float(box.conf[0]),
     #                       'bbox': box.xyxy[0].tolist()
     #                   })
            else: # model_type == 'faster_rcnn':
                detections = run_faster_rcnn_inference(frame)
            await websocket.send_json({
                "detections": detections,
                "is_compliant": True,
                "missing_ppe": [],
                "timestamp": ""
            })
    except WebSocketDisconnect:
        return
