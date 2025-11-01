import torch
import torchvision
print('Downloading pretrained weights (may take a minute)...')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
torch.save(model.state_dict(), "models/faster_rcnn.pth")
print("Saved as models/faster_rcnn.pth")
