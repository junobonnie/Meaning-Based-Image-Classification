import argparse
import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor
from tqdm import tqdm

from model import HybridClassifier
from utils import get_device

def main():
    parser = argparse.ArgumentParser(description='이미지 분류 및 정렬 (Inference & Sort)')
    parser.add_argument('--input_dir', type=str, required=True, help='분류할 이미지가 있는 폴더 경로')
    parser.add_argument('--output_dir', type=str, required=True, help='분류된 이미지를 저장할 폴더 경로')
    parser.add_argument('--checkpoint', type=str, required=True, help='학습된 모델 체크포인트 경로')
    parser.add_argument('--threshold', type=float, default=0.0, help='분류 확신도 임계값 (0.0 ~ 1.0)')
    parser.add_argument('--others_dir', type=str, default='others', help='임계값 미만인 이미지를 저장할 폴더 이름')
    args = parser.parse_args()

    device = get_device()
    print(f"사용 장치: {device}")

    # 체크포인트 로드
    print(f"체크포인트 로드 중: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except Exception as e:
        print(f"체크포인트 로드 실패: {e}")
        return

    classes = checkpoint['classes']
    mode = checkpoint.get('mode', 'hybrid')
    num_classes = len(classes)
    print(f"모델 모드: {mode}")
    print(f"클래스 목록: {classes}")

    # 모델 초기화
    model = HybridClassifier(num_classes=num_classes, mode=mode).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 전처리 정의
    mobilenet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # CLIP 프로세서 로드 (safetensors 사용)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 이미지 파일 목록 가져오기
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]
    
    print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Sorting"):
            file_path = os.path.join(args.input_dir, filename)
            
            try:
                image = Image.open(file_path).convert('RGB')
            except Exception as e:
                print(f"이미지 로드 오류 {filename}: {e}")
                continue

            # MobileNet 입력 준비
            mobilenet_input = mobilenet_transform(image).unsqueeze(0).to(device)

            # CLIP 입력 준비
            clip_input = clip_processor(images=image, return_tensors="pt")['pixel_values'].to(device)

            # 추론
            outputs = model(mobilenet_input, clip_input)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted_idx = probs.max(1)
            
            if max_prob.item() < args.threshold:
                predicted_class = args.others_dir
            else:
                predicted_class = classes[predicted_idx.item()]

            # 결과 폴더 생성 및 복사
            target_dir = os.path.join(args.output_dir, predicted_class)
            os.makedirs(target_dir, exist_ok=True)
            
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(file_path, target_path)

    print("분류 및 정렬 완료!")

if __name__ == '__main__':
    main()
