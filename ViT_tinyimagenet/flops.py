import torch
# from fvcore.nn import FlopCountAnalysis
from thop import profile

from model import VisionTransformer


def main():
    # # Self-Attention
    # a1 = Attention(dim=768, num_heads=1)
    # a1.proj = torch.nn.Identity()  # remove Wo
    # 
    # # Multi-Head Attention
    # a2 = Attention(dim=768, num_heads=8)
    
    # # [batch_size, num_tokens, total_embed_dim]
    # t = (torch.rand(1, 1024, 512),)
    # 
    # flops1 = FlopCountAnalysis(a1, t)
    # print("Self-Attention FLOPs:", flops1.total())
    # 
    # flops2 = FlopCountAnalysis(a2, t)
    # print("Multi-Head Attention FLOPs:", flops2.total())
    
    #ViT B/16
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

    input = torch.zeros((1, 3, 224, 224)).to(device)
    flops, params = profile(m.to(device), inputs=(input,))

    print("参数量：", params)
    print("FLOPS：", flops)


if __name__ == '__main__':
    main()
