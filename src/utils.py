import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """모델의 상태(체크포인트)를 파일로 저장합니다."""
    torch.save(state, filename)

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path="training_history.png"):
    """학습 및 검증 손실과 정확도 그래프를 그리고 저장합니다."""
    plt.figure(figsize=(12, 5))
    
    # 손실(Loss) 그래프
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 정확도(Accuracy) 그래프
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_device():
    """사용 가능한 경우 CUDA(GPU) 장치를, 그렇지 않으면 CPU를 반환합니다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
