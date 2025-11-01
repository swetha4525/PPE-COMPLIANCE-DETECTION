# AI-Powered Real-Time PPE Compliance Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.0%2B-blue)](https://reactjs.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time PPE (Personal Protective Equipment) compliance detection system for healthcare facilities using YOLOv8, YOLOv5, and YOLOv3. Detects masks, gowns, gloves, goggles, and face shields through webcam-based CCTV streams with instant alerts and comprehensive analytics.

## 🎯 Features

- **Real-time Detection**: Live webcam-based PPE detection at 30-60 FPS
- **Multi-Model Comparison**: Benchmark YOLOv8, YOLOv5, and YOLOv3 performance
- **Instant Alerts**: Visual and audio warnings for PPE violations
- **Compliance Dashboard**: Real-time analytics and historical trends
- **Privacy-First**: All processing happens locally, no cloud uploads
- **5 PPE Classes**: Coverall, Face Shield, Gloves, Goggles, Mask

## 📊 Model Performance (on CPPE-5 Dataset)

| Model | mAP@0.5 | Precision | Recall | FPS (GPU) | Size (MB) |
|-------|---------|-----------|--------|-----------|-----------|
| **YOLOv8n** | 82% | 0.80 | 0.78 | 180 | 6 |
| YOLOv5s | 75% | 0.73 | 0.70 | 140 | 14 |
| YOLOv3-tiny | 68% | 0.68 | 0.63 | 120 | 35 |

## 🏗️ System Architecture


## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- CUDA-compatible GPU (optional, for faster training/inference)
- Webcam or IP camera

### Backend Setup


### Frontend Setup


## 🚀 Quick Start

### 1. Train Models (Optional - Pre-trained weights available)


### 2. Run Backend Server


### 3. Run Frontend Application


### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs

## 📖 Usage Guide

### Live Detection

1. Navigate to the "Live Detection" tab
2. Grant webcam permissions
3. Select your YOLO model (YOLOv8/YOLOv5/YOLOv3)
4. Real-time bounding boxes will appear
5. Alerts trigger for missing PPE

### Dashboard Analytics

1. See compliance statistics, rate, violations and trends
2. Export logs for audit

### Model Comparison

View performance metrics and radar charts per model.

## 📁 Project Structure


## 🔧 Configuration
_Backend/app.py and train_yolov8.py have all the config options referenced above._

## 📝 License

MIT License

---
**If you find this project helpful, give it a star!**
