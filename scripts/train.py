"""사전학습 YOLO11n을 3개 부류(elephant, rhino, zebra) 데이터로 fine-tuning"""
import sys
from ultralytics import YOLO
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
name = sys.argv[2] if len(sys.argv) > 2 else 'yolo11n_wildlife3'
model = YOLO('yolo11n.pt')                 # COCO 사전학습 가중치 (전이학습)
model.train(
    data='data.yaml',
    epochs=epochs, imgsz=416, batch=16,
    device='cpu', workers=2, seed=0,
    optimizer='AdamW', lr0=0.002, cos_lr=True,
    patience=10, plots=True,
    project='runs', name=name, exist_ok=True,
)
