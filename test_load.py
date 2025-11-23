import torch
import torchvision.models as models
from transformers import CLIPModel

print("Loading MobileNet...")
try:
    models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    print("MobileNet loaded.")
except Exception as e:
    print(f"MobileNet failed: {e}")

print("Loading CLIP...")
try:
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP loaded.")
except Exception as e:
    print(f"CLIP failed: {e}")
