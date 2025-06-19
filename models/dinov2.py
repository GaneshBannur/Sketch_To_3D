from transformers import AutoImageProcessor, AutoBackbone
import torch
from torch import Tensor
from PIL.ImageFile import ImageFile
from typing import List

class DINOv2WithRegistersForInference:
    def __init__(self, hf_repo: str = "facebook/dinov2-with-registers-giant") -> None:
        self.processor = AutoImageProcessor.from_pretrained(hf_repo, use_fast=True)
        self.model = AutoBackbone.from_pretrained(hf_repo)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    @torch.no_grad()
    def embed(self, img_list: List[ImageFile]) -> Tensor:
        inputs = self.processor(img_list, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        outputs = self.model(**inputs)
        # outputs.feature_maps[0] has shape (batch_size, embed_dim, embed_height, embed_width)
        return outputs.feature_maps[0]

if __name__ == "__main__":
    dino = DINOv2WithRegistersForInference()
    from PIL import Image
    img = Image.open("test_img_0.jpg")
    dino.embed([img, img])
