# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size//2), bias=bias)

# class MeanShift(nn.Conv2d):
#     def __init__(
#         self, rgb_range,
#         rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
#         for p in self.parameters():
#             p.requires_grad = False

# class BasicBlock(nn.Sequential):
#     def __init__(
#         self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
#         bn=True, act=nn.ReLU(True)):

#         m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
#         if bn:
#             m.append(nn.BatchNorm2d(out_channels))
#         if act is not None:
#             m.append(act)

#         super(BasicBlock, self).__init__(*m)

# class ResBlock(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(n_feats, n_feats, kernel_size=kernel_size, bias=bias))
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if i == 0:
#                 m.append(act)

#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale

#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x

#         return res

# class Upsampler(nn.Sequential):
#     def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

#         m = []
#         if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(conv(n_feats, 4 * n_feats, 3, bias))
#                 m.append(nn.PixelShuffle(2))
#                 if bn:
#                     m.append(nn.BatchNorm2d(n_feats))
#                 if act == 'relu':
#                     m.append(nn.ReLU(True))
#                 elif act == 'prelu':
#                     m.append(nn.PReLU(n_feats))

#         elif scale == 3:
#             m.append(conv(n_feats, 9 * n_feats, 3, bias))
#             m.append(nn.PixelShuffle(3))
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if act == 'relu':
#                 m.append(nn.ReLU(True))
#             elif act == 'prelu':
#                 m.append(nn.PReLU(n_feats))
#         else:
#             raise NotImplementedError

#         super(Upsampler, self).__init__(*m)





# 导入数学运算库，用于后续的数学计算
import math
# 导入PyTorch深度学习框架的核心模块
import torch
# 导入PyTorch的神经网络模块，用于构建网络层
import torch.nn as nn
# 导入PyTorch的函数式接口，提供各种神经网络操作函数
import torch.nn.functional as F

# 定义一个默认的卷积层创建函数，简化卷积层的初始化过程
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    # 创建并返回一个2D卷积层
    # padding设置为kernel_size//2，确保在步长为1时输入输出特征图尺寸相同
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

# 定义均值偏移层，用于图像的均值归一化或反归一化操作
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,  # RGB值的范围（例如255）
        rgb_mean=(0.4488, 0.4371, 0.4040),  # 图像数据集的RGB通道均值
        rgb_std=(1.0, 1.0, 1.0),  # 图像数据集的RGB通道标准差
        sign=-1):  # 符号，-1用于归一化（减去均值），1用于反归一化（加上均值）

        # 初始化父类nn.Conv2d，创建1x1的3输入3输出卷积层
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        # 将标准差转换为张量
        std = torch.Tensor(rgb_std)
        # 初始化权重：单位矩阵除以标准差，实现标准化（x/σ）
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        # 初始化偏置：符号×RGB范围×均值/标准差，实现均值偏移（-μ/σ或+μ/σ）
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        # 冻结所有参数，不参与训练（均值和标准差是固定的先验知识）
        for p in self.parameters():
            p.requires_grad = False

# 定义基础网络块，由卷积层、可选的批归一化层和激活函数组成
class BasicBlock(nn.Sequential):
    def __init__(
        self, conv,  # 卷积函数（如default_conv）
        in_channels,  # 输入特征图通道数
        out_channels,  # 输出特征图通道数
        kernel_size,  # 卷积核大小
        stride=1,  # 卷积步长
        bias=False,  # 卷积层是否使用偏置
        bn=True,  # 是否添加批归一化层
        act=nn.ReLU(True)):  # 激活函数

        # 创建网络层列表
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            # 如果需要批归一化，添加批归一化层
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            # 如果需要激活函数，添加激活函数
            m.append(act)

        # 调用父类nn.Sequential的构造函数，将网络层组合成序列
        super(BasicBlock, self).__init__(*m)

# 定义残差块，包含两个卷积层和残差连接
class ResBlock(nn.Module):
    def __init__(
        self, conv,  # 卷积函数
        n_feats,  # 特征图通道数（输入输出通道数相同）
        kernel_size,  # 卷积核大小
        bias=True,  # 卷积层是否使用偏置
        bn=False,  # 是否添加批归一化层
        act=nn.ReLU(True),  # 激活函数
        res_scale=1):  # 残差缩放因子

        super(ResBlock, self).__init__()
        m = []  # 存储网络层的列表
        for i in range(2):  # 创建两个卷积层
            # 添加卷积层，保持通道数不变
            m.append(conv(n_feats, n_feats, kernel_size=kernel_size, bias=bias))
            if bn:
                # 如果需要批归一化，添加批归一化层
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                # 只在第一个卷积层后添加激活函数
                m.append(act)

        # 将网络层组合成序列，作为残差块的主体
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale  # 残差缩放因子

    def forward(self, x):
        # 计算残差：通过网络主体处理后乘以缩放因子
        res = self.body(x).mul(self.res_scale)
        # 残差连接：残差与输入相加
        res += x

        return res

# 定义上采样模块，用于提升特征图分辨率（超分辨率核心组件）
class Upsampler(nn.Sequential):
    def __init__(self, conv,  # 卷积函数
                 scale,  # 缩放倍数（如2、3、4等）
                 n_feats,  # 输入特征图通道数
                 bn=False,  # 是否添加批归一化层
                 act=False,  # 激活函数类型（'relu'、'prelu'或False）
                 bias=True):  # 卷积层是否使用偏置

        m = []  # 存储网络层的列表
        # 检查缩放倍数是否为2的幂次方（如2、4、8等）
        if (scale & (scale - 1)) == 0:    
            # 计算需要多少次2倍上采样（例如4倍需要2次）
            for _ in range(int(math.log(scale, 2))):
                # 卷积层将通道数扩展为4倍（为PixelShuffle做准备）
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                # PixelShuffle(2)将特征图尺寸放大2倍，通道数缩小4倍
                m.append(nn.PixelShuffle(2))
                if bn:
                    # 如果需要批归一化，添加批归一化层
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    # 添加ReLU激活函数
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    # 添加PReLU激活函数
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:  # 处理3倍缩放的情况
            # 卷积层将通道数扩展为9倍（为PixelShuffle做准备）
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            # PixelShuffle(3)将特征图尺寸放大3倍，通道数缩小9倍
            m.append(nn.PixelShuffle(3))
            if bn:
                # 如果需要批归一化，添加批归一化层
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                # 添加ReLU激活函数
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                # 添加PReLU激活函数
                m.append(nn.PReLU(n_feats))
        else:
            # 不支持其他缩放倍数
            raise NotImplementedError

        # 调用父类nn.Sequential的构造函数，将网络层组合成序列
        super(Upsampler, self).__init__(*m)