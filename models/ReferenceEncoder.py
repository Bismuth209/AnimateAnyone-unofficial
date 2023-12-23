import torch
import lovely_tensors
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPImageProcessor
from transformers import logging

lovely_tensors.monkey_patch()
logging.set_verbosity_warning()
logging.set_verbosity_error()

# https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train_plus.py#L49


class ReferenceEncoder(nn.Module):
    def __init__(self, model_path="checkpoints/clip-vit-base-patch32"):
        super(ReferenceEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(model_path, local_files_only=True)
        self.processor = CLIPImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input):
        if isinstance(input, Image.Image):
            pixel_values = self.processor.preprocess(input, return_tensors="pt")[
                "pixel_values"
            ]
        elif isinstance(input, torch.Tensor):
            pixel_values = input
        outputs = self.model(pixel_values)
        pooled_output = outputs.pooler_output
        return pooled_output


if __name__ == "__main__":
    # example
    image_path = "/home/ubuntu/data/animate-anyone/TikTok_dataset/00001/images/0001.png"
    image = Image.open(image_path).convert("RGB")

    model = ReferenceEncoder()
    pooled_output = model(image)

    print(f"Pooled Output Size: {pooled_output.size()}")
    # tensor[1, 3, 224, 224] n=150528 (0.6Mb) x∈[-1.752, 1.930] μ=0.067 σ=1.109
    # Pooled Output Size: torch.Size([bs, 768])
