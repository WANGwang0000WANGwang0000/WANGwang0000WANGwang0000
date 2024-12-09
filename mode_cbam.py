import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        #yprint(x.shape)
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result






import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.ops import StochasticDepth

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


class SELayer(nn.Module):   #定义得类用于实现SE模块，SE模块是一种轻量级的注意力机制，可以增强深度神经网络的表征能力
    def __init__(self, channel, reduction=16):   #reduction：代表 SE 模块中 MLP 中间层的输出通道数与输入通道数之比，
        super(SELayer, self).__init__()          #使用两个全连接层和一个 Sigmoid 激活函数构建一个轻量级的多层感知机 (MLP)，用于计算通道间的权重系数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   #创建一个自适应的平均池化层，使得输入特征图可以在任何大小的情况下都被池化成一个固定大小的张量
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): #forward 方法接受一个四维的输入张量 x，通过平均池化和 MLP 计算出输入通道的权重系数，并将其与输入张量进行元素乘法，最后返回加权后的特征图
        b, c, _, _ = x.size()   # (1, 1) 表示在每个通道上对整个特征图进行平均池化，并将其池化成一个 (1, 1) 的张量
        y = self.avg_pool(x).view(b, c)  #对输入张量 x 进行自适应平均池化，将其降维为 (B, C, 1, 1) 的张量
        y = self.fc(y).view(b, c, 1, 1)  #self.fc(y) 对输入张量 y 进行 MLP 计算，并将其输出张量的形状从 (B, C) 转换为 (B, C//r, 1, 1)
        return x * y.expand_as(x)#r 是 reduction 参数，代表 MLP 中间层的输出通道数与输入通道数之比。具体地，self.fc 中的两个全连接层和 ReLU 激活函数构成了一个两层的 MLP，用于计算每个通道的权重系数。
#该步骤的主要作用是将 MLP 的输出结果转换为与原始输入特征图相同的维度，以便进行元素乘法操作。

class DotProductSelfAttention(nn.Module):      #代码定义了一个名为 DotProductSelfAttention 的类，用于实现基于点积的自注意力机制。
    def __init__(self, input_dim):
        super(DotProductSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)  #首先使用 nn.LayerNorm 对输入张量进行归一化操作，以提高模型的训练稳定性
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.norm(x)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
##########这行代码计算了一个用于缩放自注意力分数的标量值。它使用了输入维度 self.input_dim 的平方根的平方根的倒数，
        scale = 1 / math.sqrt(math.sqrt(self.input_dim))  #这种缩放是为了避免自注意力分数过大或过小的问题。较大的分数可能导致梯度爆炸，而较小的分数可能导致梯度消失
        #torch.matmul 是 PyTorch 提供的矩阵相乘的函数，它接受两个张量作为输入，并返回它们的矩阵乘积。在这里，query 和 key 分别表示查询和键的表示张量，它们的形状通常是 (B, L, D)，其中 B 表示批次大小，L 表示序列长度，D 表示隐藏维度。
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale  #torch.matmul 函数计算查询（query）和键（key）的点积，并乘以缩放系数 scale
        attention_weights = torch.softmax(scores, dim=-1)   # key.transpose(-2, -1) 实现，其中 -2 和 -1 表示倒数第二个和倒数第一个维度进行交换操作
#计算查询和键的点积，得到一个形状为 (B, L, L) 的张量，其中每个元素表示查询和键之间的相关性。然后，将缩放系数 scale 乘以点积结果，以便在 softmax 操作之前对自注意力分数进行缩放处理。这有助于控制分数的范围，使其更适合应用 softmax 函数，从而生成注意力权重。
        attended_values = torch.matmul(attention_weights, value)
        output = attended_values + x  #attended_values 是通过注意力权重对值进行加权求和得到的表示。它是一个形状为 (B, L, D) 的张量，其中 B 表示批次大小，L 表示序列长度，D 表示隐藏维度
        return output, attention_weights
#x 是原始的输入张量，也是一个形状为 (B, L, D) 的张量。通过将 attended_values 与 x 相加，可以将自注意力机制整合的信息与原始输入的信息进行融合，从而得到最终的输出表示。
#这样做的目的是将自注意力机制提取的关键信息与原始输入的信息相结合，充分利用两者之间的互补性，从而得到更丰富、更准确的表示。最终的输出可以用于后续的任务，如分类、生成等。
class LayerNorm(nn.Module):  #为了对输入进行标准化处理
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):#定义的这个模块包含了基本的残差结构
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv  #的深度卷积层，用于对输入进行特征提取
        self.norm = LayerNorm(dim, eps=1e-6)#norm 是一个层归一化层，用于对输入进行标准化；
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers   #pwconv1 和 pwconv2 分别是两个线性层，用于进行维度变换
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.stochastic_depth = StochasticDepth(drop_path, "row")     #StochasticDepth 层，用于实现残差连接中的随机深度

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)  #permute 函数对张量 x 进行维度的置换操作。在这里，(0, 2, 3, 1) 表示将原始张量的维度顺序变为 (0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.stochastic_depth(x)
        return x


class EmoNeXt(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000,  #ConvNetXt 模块的深度列表，包含每个阶段中 ConvNeXt 模块的数量，默认为 [3, 3, 9, 3]。
                 depths=None, dims=None, drop_path_rate=0.,    #ConvNeXt 模块的维度列表，包含每个阶段中 ConvNeXt 模块的输出通道数，默认为 [96, 192, 384, 768]
                 layer_scale_init_value=1e-6,   #drop_path_rate：随机深度剪枝（Stochastic Depth）的丢弃率，默认为 0
                 ):              #layer_scale_init_value：ConvNeXt 模块中缩放参数的初始值，默认为 1e-6
        super().__init__()

        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]

        # Spatial transformer localization-network
        self.localization = nn.Sequential(    #是一个神经网络模型中的一部分，用于实现图像的定位功能
            nn.Conv2d(3, 8, kernel_size=7),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(               #定义了一个全连接网络（nn.Sequential），用于定位任务（Localization Task）
            #将输入特征图展平为一个向量，并将其映射到大小为 32 的特征向量这是第一个线性层
            nn.Linear(10 * 52 * 52, 32),   #它接受输入大小为 10 × 52 × 52 的特征图。这里的 10 表示特征图的通道数，52 × 52 表示特征图的空间尺寸。
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)#通过 nn.Linear(32, 3 * 2) 将特征向量映射到大小为 3 × 2 的输出向量
        )
#输入的特征图通过一个卷积层和一个层归一化层进行降采样，输出的特征图尺寸变为原来的1/4，并且通道数变为dims[0]。这通常用于在深层网络中逐渐减小特征图的尺寸和通道数，以提取更高层次的特征。
        self. downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)#这段代码是一个神经网络的下采样部分。其中，包含了一个卷积层和SENet中的SE模块，用于增强网络的表达能力和减少参数量
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                SELayer(dims[i + 1])
            )
            self.downsample_layers.append(downsample_layer)#在某个类或函数中添加一个下采样层（downsample layer）到self.downsample_layers列表中。

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0   #这段代码是一个神经网络模型的初始化部分，其中定义了4个特征分辨率阶段（stages），每个阶段由多个残差块（residual blocks）组成。
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.attention = DotProductSelfAttention(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.cbam =CBAM(30)
#这段代码实现了一个空间变换网络（Spatial Transformer Network，STN）的功能
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward_features(self, x):#通过一个循环，依次对输入x进行下采样和特征提取。下采样操作使用self.downsample_layers[i]来降低输入的空间分辨率。然后，通过self.stages[i]对下采样后的特征进行进一步处理。
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)   #用x.mean([-2, -1])计算全局平均池化，将输入从(N, C, H, W)维度降至(N, C)维度。最后，返回处理后的特征。
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, labels=None):
        x = self.stn(x)         #输入图像进入了stn网络了
        #print(x.shape)
        x = x.repeat(1, 10, 1, 1)
        #print(x.shape)
        x =self.cbam(x)
        #print(x.shape)
        x = x.narrow(1, 0, 3)
        #print(x.shape)
        x = self.forward_features(x)
        _, weights = self.attention(x)     #在进入分类头之前加了一个点击注意力
        logits = self.head(x)

        if labels is not None:
            mean_attention_weight = torch.mean(weights)
            attention_loss = torch.mean((weights - mean_attention_weight) ** 2)

            loss = F.cross_entropy(logits, labels, label_smoothing=0.2) + attention_loss
            return torch.argmax(logits, dim=1), logits, loss

        return torch.argmax(logits, dim=1), logits


def get_model(num_classes, model_size='tiny', in_22k=False):
    if model_size == 'tiny':
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
    elif model_size == 'small':
        depths = [3, 3, 27, 3]
        dims = [96, 192, 384, 768]
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
    elif model_size == 'base':
        depths = [3, 3, 27, 3]
        dims = [128, 256, 512, 1024]
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
    elif model_size == 'large':
        depths = [3, 3, 27, 3]
        dims = [192, 384, 768, 1536]
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
    else:
        depths = [3, 3, 27, 3]
        dims = [256, 512, 1024, 2048]
        url = model_urls['convnext_xlarge_22k']

    default_num_classes = 1000
    if in_22k:
        default_num_classes = 21841

    net = EmoNeXt(
        depths=depths,
        dims=dims,
        num_classes=default_num_classes,
        drop_path_rate=0.1
    )

    state_dict = load_state_dict_from_url(url=url)
    net.load_state_dict(state_dict['model'], strict=False)
    net.head = nn.Linear(dims[-1], num_classes)

    return net
if __name__ == '__main__':

    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Using device:', device)

    model = get_model(7,'tiny').to(device)
    #print(model)
    input_size = (3, 224, 224)
    #
    summary(model, input_size=input_size)

    # 测试一张48*48的随机图像
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x)
    print(output)
