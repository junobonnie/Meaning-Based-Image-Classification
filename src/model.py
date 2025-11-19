import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPModel

class HybridClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, dropout=0.5, mode='hybrid'):
        super(HybridClassifier, self).__init__()
        self.mode = mode
        
        # 1. MobileNetV3 (시각적 특징)
        if self.mode in ['hybrid', 'mobilenet']:
            # 사전 학습된 MobileNetV3 Large 로드
            self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            self.mobilenet_features_dim = 960 
            
            # 초기에는 백본 모델 고정
            for param in self.mobilenet.parameters():
                param.requires_grad = False
        else:
            self.mobilenet = None
            self.mobilenet_features_dim = 0

        # 2. CLIP (의미적 특징)
        if self.mode in ['hybrid', 'clip']:
            # 사전 학습된 CLIP 모델 로드
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_features_dim = 512 
            
            # 초기에는 백본 모델 고정
            for param in self.clip.parameters():
                param.requires_grad = False
        else:
            self.clip = None
            self.clip_features_dim = 0
            
        # 3. 융합 및 분류 헤드 (FNN)
        # 모드에 따라 입력 차원 결정
        if self.mode == 'hybrid':
            self.fusion_dim = self.mobilenet_features_dim + self.clip_features_dim
        elif self.mode == 'mobilenet':
            self.fusion_dim = self.mobilenet_features_dim
        elif self.mode == 'clip':
            self.fusion_dim = self.clip_features_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, mobilenet_input, clip_input):
        features_list = []
        
        # MobileNet 특징 추출
        if self.mode in ['hybrid', 'mobilenet']:
            x_mobilenet = self.mobilenet.features(mobilenet_input)
            x_mobilenet = self.mobilenet.avgpool(x_mobilenet) 
            x_mobilenet = torch.flatten(x_mobilenet, 1) 
            features_list.append(x_mobilenet)
        
        # CLIP 특징 추출
        if self.mode in ['hybrid', 'clip']:
            # CLIP 모드일 때 clip_input이 None이 아닌지 확인
            if clip_input is not None:
                x_clip = self.clip.get_image_features(clip_input)
                features_list.append(x_clip)
        
        # 특징 결합
        if len(features_list) > 1:
            x_combined = torch.cat(features_list, dim=1)
        else:
            x_combined = features_list[0]
        
        # 분류 수행
        output = self.classifier(x_combined)
        
        return output
    
    def unfreeze_backbones(self, unfreeze_mobilenet=True, unfreeze_clip=False):
        """백본 모델의 파라미터 고정을 해제하여 미세 조정을 가능하게 합니다."""
        if self.mobilenet and unfreeze_mobilenet:
            for param in self.mobilenet.parameters():
                param.requires_grad = True
        if self.clip and unfreeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = True
