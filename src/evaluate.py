import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPProcessor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataset import HybridDataset
from model import HybridClassifier
from utils import get_device

def evaluate(model, loader, device, classes):
    """모델을 평가하고 분류 보고서와 혼동 행렬을 출력합니다."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for mobilenet_input, clip_input, targets in loader:
            mobilenet_input = mobilenet_input.to(device)
            clip_input = clip_input.to(device)
            targets = targets.to(device)
            
            outputs = model(mobilenet_input, clip_input)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    print("분류 보고서 (Classification Report):")
    print(classification_report(all_targets, all_preds, target_names=classes))
    
    # 혼동 행렬 시각화 및 저장
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('예측값 (Predicted)')
    plt.ylabel('실제값 (True)')
    plt.title('혼동 행렬 (Confusion Matrix)')
    plt.savefig('confusion_matrix.png')
    print("혼동 행렬이 confusion_matrix.png 파일로 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='하이브리드 이미지 분류기 평가')
    parser.add_argument('--data_dir', type=str, required=True, help='데이터셋 루트 경로')
    parser.add_argument('--checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    args = parser.parse_args()
    
    device = get_device()
    print(f"사용 장치: {device}")
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint['classes']
    mode = checkpoint.get('mode', 'hybrid') # 이전 체크포인트 호환성을 위해 기본값 'hybrid'
    num_classes = len(classes)
    
    print(f"모델 모드: {mode}")
    
    # 변환 (Transforms)
    mobilenet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    
    # 데이터셋 로드
    dataset = HybridDataset(args.data_dir, transform=mobilenet_transform, clip_processor=clip_processor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 모델 초기화 및 가중치 로드
    model = HybridClassifier(num_classes=num_classes, mode=mode).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    
    evaluate(model, loader, device, classes)

if __name__ == '__main__':
    main()
