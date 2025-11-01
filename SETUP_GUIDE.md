# Detailed Setup Guide

## System Requirements

### Minimum Requirements
- **CPU**: Intel Core i5 or equivalent
- **RAM**: 8 GB
- **Storage**: 10 GB
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+

### Recommended Requirements
- **CPU**: Intel Core i7 or equivalent
- **GPU**: NVIDIA GTX 1660+ (6GB+ VRAM)
- **RAM**: 16 GB
- **Storage**: 20 GB SSD
- **OS**: Ubuntu 22.04 LTS

## Installation Steps

### 1. Python Environment

python --version
python -m venv venv
source venv/bin/activate # Or venv\Scripts\activate on Windows
pip install --upgrade pip

### 2. Backend Dependencies

cd backend
pip install -r requirements.txt

### 3. Pretrained Weights (Download or train yourself)
Create `models/` and place weights files (`yolov8.pt`, etc).

### 4. Prepare CPPE-5 Dataset

python prepare_cppe5_dataset.py

### 5. (Optional) Train YOLOv8

python train_yolov8.py

### 6. Frontend Setup

cd frontend
npm install
npx tailwindcss init -p

### 7. Launch
#### Backend

cd backend
python app.py

#### Frontend

cd frontend
npm start

### 8. Docker
Use `docker-compose.yml` for production deployment, see documentation above.

## Troubleshooting

See README.md "Troubleshooting" section for common errors (CUDA, webcam permissions, etc).

## Next Steps
- Test live detection
- Review dashboard analytics
- Compare model performance
- Deploy to production

---
For more help, see [API Documentation](http://localhost:8000/docs)
