from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.__floor__()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

#以下注释均以224×224×3大小的图片，patch大小为16×16为前提（ViT B/16）
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm =norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H==self.img_size[0] and W==self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        #卷积之后：【B,3,224,224】->【B,768,14,14】
        #flatten: 将【B,C,H,W】后两个维度即H和W展开成一维序列【B,C,HW】
        #transpose: 将一维序列的C与HW换位置成【B,HW,C】
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,#输入token的维度dim，即768
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ation=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads#每一个头的维度
        self.scale = qk_scale or head_dim ** -0.5#这个参数就是注意力计算公式中的1/根号（dk）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)#这里只采用一个全连接层而不是三个全连接层来生成qkv，有助于并行化，加快训练速度
        self.attn_drop = nn.Dropout(attn_drop_ation)
        self.proj = nn.Linear(dim,dim)#在多头注意力的最后会将所有头得到的结果进行拼接，再通过一个W进行一个映射，Cancat(head1,head2,...)*W，其中headi=Attention(Q,K,V)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        #[batch_Size, num_patches + 1, total_embed_dim]，即[batchsize, 196+1, 768]
        B, N, C = x.shape

        #qkv():->[batch_Size, num_patches + 1, total_embed_dim * 3]
        #reshape:->[batch_Size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        #permute:->[3, batch_size, num_heads, num_patches+1, embed_dim_per_head]
        #q, k, v:->[batch_size, num_heads, num_patches+1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        #transpose:->[batch_size, num_heads, embed_dim_per_head, num_patches+1]
        #@：就是矩阵相乘multiply->[batch_size, num_heads, num_patches+1, num_patches+1],高维的Tensor相乘维度要求:"除了最后2维"的其他维度尺寸必须完全对应相等；满足矩阵相乘的尺寸规律.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)#即在每一行上softmax
        attn = self.attn_drop(attn)

        #@:->[batch_size, num_heads, num_patches+1, embed_dim_per_head]
        #transpose:->[batch_size, num_patches+1, num_heads, embed_dim_per_head]
        #reshape:->[batch_size, num_patches+1, total_embed_dim]
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class InterAttention(nn.Module):
    def __init__(self,
                 dim,#输入token的维度dim，即768
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ation=0.,
                 proj_drop_ratio=0.):
        super(InterAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads#每一个头的维度
        self.scale = qk_scale or head_dim ** -0.5#这个参数就是注意力计算公式中的1/根号（dk）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)#这里只采用一个全连接层而不是三个全连接层来生成qkv，有助于并行化，加快训练速度
        self.attn_drop = nn.Dropout(attn_drop_ation)
        self.proj = nn.Linear(dim,dim)#在多头注意力的最后会将所有头得到的结果进行拼接，再通过一个W进行一个映射，Cancat(head1,head2,...)*W，其中headi=Attention(Q,K,V)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        #[batch_Size, num_patches + 1, total_embed_dim]，即[batchsize, 196+1, 768]
        B, N, C = x.shape

        #qkv():->[batch_Size, num_patches + 1, total_embed_dim * 3]
        #reshape:->[batch_Size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        #permute:->[3, batch_size, num_heads, num_patches+1, embed_dim_per_head]
        #q, k, v:->[batch_size, num_heads, num_patches+1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        #transpose:->[batch_size, num_heads, embed_dim_per_head, num_patches+1]
        #@：就是矩阵相乘multiply->[batch_size, num_heads, num_patches+1, num_patches+1],高维的Tensor相乘维度要求:"除了最后2维"的其他维度尺寸必须完全对应相等；满足矩阵相乘的尺寸规律.
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)#即在每一行上softmax
        attn = self.attn_drop(attn)

        #@:->[batch_size, num_heads, num_patches+1, embed_dim_per_head]
        #transpose:->[batch_size, num_patches+1, num_heads, embed_dim_per_head]
        #reshape:->[batch_size, num_patches+1, total_embed_dim]
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ation=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        #原论文在多头注意力和mlp之后采用的是dropout，这里采用的是droppath
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class InterEncoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(InterEncoderBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = InterAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ation=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        #原论文在多头注意力和mlp之后采用的是dropout，这里采用的是droppath
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_raito=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        """
        :param img_size(int, tuple):input image size
        :param patch_size(int ,tuple):patch size
        :param in_c(int):number of input channels
        :param num_classes(int):number of classes for classification head
        :param embed_dim(int):embedding dimension
        :param depth(int):depth of the transformer(number of encoder layers)
        :param num_heads(int):number of attention heads
        :param mlp_ratio(int):ratio of mlp hidden dim to embedding dim
        :param qkv_bias(bool):enable bias for qkv if True
        :param qk_scale(float):override default qk scale of head_dim ** -0.5 if set
        :param representation_size(Optional[int]):最后分类的MLP head中的pre logits的相关参数
        :param distilled(bool):用于搭建DeiT模型所使用的
        :param drop_ratio(float):dropout rate
        :param attn_drop_ratio(float):attention dropout rate
        :param drop_path_raito(float):stochastic depth rate
        :param embed_layer(nn.Module):patch embedding layer
        :param norm_layer(nn.Module):normalization layer
        :param act_layer:activation function
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        #构建一个等差序列，从0到depth_path_ratio，在堆叠的encoder layers中使用的dropout ratio是一个递增的等差序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_raito, depth)]
        #HybridEncoder(Origin, Inter)
        # blocks = []
        # for i in range(depth):
        #     if i % 2 == 0:
        #         blocks.append(EncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                                    qk_scale=qk_scale, drop_ratio=drop_ratio,
        #                                    attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #                                    norm_layer=norm_layer, act_layer=act_layer))
        #     else:
        #         blocks.append(
        #             InterEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                               qk_scale=qk_scale, drop_ratio=drop_ratio,
        #                               attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], norm_layer=norm_layer,
        #                               act_layer=act_layer))
        # self.blocks = nn.Sequential(*blocks)

        #HybridEncoder2(Inter, Origin)
        # blocks = []
        # for i in range(depth):
        #     if i % 2 == 0:
        #         blocks.append(
        #             InterEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                                    qk_scale=qk_scale, drop_ratio=drop_ratio,
        #                                    attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #                                    norm_layer=norm_layer, act_layer=act_layer))
        #     else:
        #         blocks.append(
        #             EncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                               qk_scale=qk_scale, drop_ratio=drop_ratio,
        #                               attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], norm_layer=norm_layer,
        #                               act_layer=act_layer))
        # self.blocks = nn.Sequential(*blocks)

        #InterEncoder(All Inter)
        # self.blocks = nn.Sequential(*[
        #     InterEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                                        qk_scale=qk_scale, drop_ratio=drop_ratio,
        #                                        attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #                                        norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])

        #Origin Encoder
        self.blocks = nn.Sequential(*[
            EncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale, drop_ratio=drop_ratio,
                                               attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                                               norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        #representation layer:pre-logits 层通常被用来对特征进行降维或者增加特征的非线性表示，以更好地适应分类任务
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                                                ("fc", nn.Linear(embed_dim, representation_size)),
                                                ("act", nn.Tanh())
            ]))#OrderedDict：指定了一个有序的字典，定义了 nn.Sequential 中每个子模块的名称和对应的神经网络层，保持有序并且能够通过键名来访问子模块
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        #分类头classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity

        #权重初始化weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)#trunc_normal_:截断的正态分布（truncated normal distribution）初始化给定的张量,以避免取值超出特定的范围。
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        #[B,C,H,W]->[B,num_patches,embed_dim]
        x = self.patch_embed(x) # [B,196,768]
        #[1,1,768]->[B,1,768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)#因为cls_token初始化的时候第一个维度即batchsize为1，需要扩展到和x一样
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)#[B,197,768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)#如果distilled为True，[B,198,768]

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768 if has_logits else None,
        num_classes=num_classes
    )
    return model

def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768 if has_logits else None,
        num_classes=num_classes
    )
    return model

def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024 if has_logits else None,
        num_classes=num_classes
    )
    return model

def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024 if has_logits else None,
        num_classes=num_classes
    )
    return model

def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        representation_size=1280 if has_logits else None,
        num_classes=num_classes
    )
    return model