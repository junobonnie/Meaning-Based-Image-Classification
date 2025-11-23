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

class EarlyStopping:
    """검증 손실이 개선되지 않으면 학습을 조기 종료합니다."""
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): 개선이 없는 에폭을 얼마나 기다릴지 설정
            min_delta (float): 개선이라고 판단할 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
