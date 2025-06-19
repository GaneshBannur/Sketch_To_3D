from typing import List, Tuple
from transformers import AutoImageProcessor
from PIL import Image
from PIL.ImageFile import ImageFile
from collections import defaultdict

class DINOCollator:
    def __init__(self, dino_processor: AutoImageProcessor) -> None:
        self.dino_processor = dino_processor

    def collate_for_dino(self, img_paths: List[str]) -> Tuple[List[ImageFile], int, int]:
        batch_imgs = defaultdict(list)
        for nth_imgs in img_paths:
            for i, p in enumerate(nth_imgs):
                batch_imgs[i].append(Image.open(p))
        
        batch_size = len(batch_imgs)
        num_img_per_item = len(batch_imgs[0])
        imgs = []
        for e in batch_imgs.values():
            imgs.extend(e)

        return imgs, batch_size, num_img_per_item
