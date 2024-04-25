import torch
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.yaml")
    results = model.train(data="config.yaml")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
