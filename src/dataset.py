import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HybridDataset(Dataset):
    def __init__(self, root_dir, transform=None, clip_processor=None):
        """
        Args:
            root_dir (string): 모든 이미지가 있는 디렉토리 경로.
            transform (callable, optional): 샘플에 적용할 선택적 변환 (MobileNet용).
            clip_processor (callable, optional): CLIP 모델을 위한 전처리기.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.clip_processor = clip_processor
        # 하위 디렉토리 이름을 클래스 이름으로 사용 (알파벳 순 정렬)
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._make_dataset()

    def _make_dataset(self):
        """이미지 경로와 레이블을 리스트로 생성합니다."""
        images = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            for root, _, fnames in os.walk(class_dir):
                for fname in fnames:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[target_class])
                        images.append(item)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, target = self.images[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 오류 {path}: {e}")
            # 오류 발생 시 검은 이미지 반환 (충돌 방지)
            image = Image.new('RGB', (224, 224))

        # MobileNet 입력 준비
        mobilenet_input = image
        if self.transform:
            mobilenet_input = self.transform(image)

        # CLIP 입력 준비
        clip_input = image
        if self.clip_processor:
            # CLIP 프로세서는 'pixel_values'를 포함한 딕셔너리를 반환합니다.
            # 배치 차원을 제거하기 위해 squeeze(0)를 사용합니다.
            clip_input = self.clip_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        return mobilenet_input, clip_input, target
