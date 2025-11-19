import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import CLIPProcessor

from dataset import HybridDataset
from model import HybridClassifier
from utils import get_device, plot_training_history, save_checkpoint
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 에폭 동안 모델을 학습합니다."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for mobilenet_input, clip_input, targets in tqdm(loader, desc="Training"):
        mobilenet_input = mobilenet_input.to(device)
        clip_input = clip_input.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(mobilenet_input, clip_input)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * mobilenet_input.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """검증 데이터셋으로 모델을 평가합니다."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mobilenet_input, clip_input, targets in tqdm(loader, desc="Validation"):
            mobilenet_input = mobilenet_input.to(device)
            clip_input = clip_input.to(device)
            targets = targets.to(device)
            
            outputs = model(mobilenet_input, clip_input)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * mobilenet_input.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description='하이브리드 이미지 분류기 학습')
    parser.add_argument('--data_dir', type=str, required=True, help='데이터셋 루트 경로')
    parser.add_argument('--epochs', type=int, default=10, help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='체크포인트 저장 디렉토리')
    parser.add_argument('--mode', type=str, default='hybrid', choices=['hybrid', 'mobilenet', 'clip'], help='모델 모드: hybrid, mobilenet, clip')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"사용 장치: {device}")
    print(f"선택된 모드: {args.mode}")
    
    # 변환 (Transforms)
    mobilenet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 데이터셋 로드
    full_dataset = HybridDataset(args.data_dir, transform=mobilenet_transform, clip_processor=clip_processor)
    num_classes = len(full_dataset.classes)
    print(f"발견된 클래스 {num_classes}개: {full_dataset.classes}")
    
    # 학습/검증 데이터 분할 (80% 학습, 20% 검증)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # Windows 호환성을 위해 num_workers=0
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 모델 초기화
    model = HybridClassifier(num_classes=num_classes, mode=args.mode).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr) # 초기에는 분류기만 최적화
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(args.epochs):
        print(f"에폭 {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"학습 손실: {train_loss:.4f}, 학습 정확도: {train_acc:.4f}")
        print(f"검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.4f}")
        
        # 체크포인트 저장
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'classes': full_dataset.classes,
            'mode': args.mode
        }, filename=os.path.join(args.output_dir, f'checkpoint_{args.mode}_epoch_{epoch+1}.pth.tar'))
        
    # 학습 기록 그래프 저장
    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=os.path.join(args.output_dir, f'history_{args.mode}.png'))
    print("학습 완료.")

if __name__ == '__main__':
    main()
