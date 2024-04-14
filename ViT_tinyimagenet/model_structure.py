import torch
from torchsummary import summary

from model import VisionTransformer

def main():
        m = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768,
        num_classes=21843
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        m.to(device)
        summary(m, (3, 224, 224))  # 输出网络结构

if __name__ == '__main__':
    main()