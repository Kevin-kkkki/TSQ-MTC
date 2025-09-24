# 导入所需模块和函数
from model import common_qipt
from model import common
from model import Quant
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange  # 用于张量维度重排的工具
import copy

# 导入自定义的量化相关模块
from .lsq_plus import *
from ._quan_base_plus import *

# 导入注意力层模块（上一个代码文件定义的内容）
from .attention_layer import *


# 创建模型的函数
def make_model(args, parent=False):
    return ipt(args)

# 定义ipt模型类（主模型类）
class ipt(nn.Module):
    def __init__(self, args, conv=Conv2dLSQ):
        super(ipt, self).__init__()
        
        self.scale_idx = 0  # 缩放因子索引，用于多尺度处理
        
        self.args = args  # 存储模型参数
        nbits = 4  # 量化位数
        
        n_feats = args.n_feats  # 特征图数量
        kernel_size = 3  # 卷积核大小
        act = nn.ReLU(True)  # 激活函数

        # 图像均值偏移层（用于预处理和后处理）
        self.sub_mean = common_qipt.MeanShift(args.rgb_range)  # 减去均值
        self.add_mean = common_qipt.MeanShift(args.rgb_range, sign=1)  # 加上均值

        # 头部网络：用于特征提取
        # 为每个缩放因子创建一个分支
        self.head = nn.ModuleList([
            nn.Sequential(
                # 初始卷积层，将输入通道数转换为特征通道数
                common_qipt.default_conv(args.n_colors, n_feats, kernel_size),
                # 残差块，使用量化卷积
                common_qipt.ResBlock(conv, n_feats, 5, act=act, nbits=nbits),
                common_qipt.ResBlock(conv, n_feats, 5, act=act, nbits=nbits)
            ) for _ in args.scale
        ])

        # 主体网络：视觉Transformer
        self.body = VisionTransformer(
            img_dim=args.patch_size,  # 图像维度
            patch_dim=args.patch_dim,  # 补丁维度
            num_channels=n_feats,  # 通道数
            embedding_dim=n_feats*args.patch_dim*args.patch_dim,  # 嵌入维度
            num_heads=args.num_heads,  # 注意力头数
            num_layers=args.num_layers,  # Transformer层数
            hidden_dim=n_feats*args.patch_dim*args.patch_dim*4,  # 隐藏层维度
            num_queries=args.num_queries,  # 查询数量
            dropout_rate=args.dropout_rate,  # dropout率
            mlp=args.no_mlp,  # 是否使用MLP
            pos_every=args.pos_every,  # 是否每层都使用位置编码
            no_pos=args.no_pos,  # 是否不使用位置编码
            no_norm=args.no_norm,  # 是否不使用归一化层
            nbits=nbits  # 量化位数
        )

        # 尾部网络：用于上采样和输出
        # 为每个缩放因子创建一个分支
        self.tail = nn.ModuleList([
            nn.Sequential(
                # 上采样层
                common_qipt.Upsampler(common_qipt.default_conv, s, n_feats, act=False),
                # 输出卷积层，将特征通道数转换为输出图像通道数
                common_qipt.default_conv(n_feats, args.n_colors, kernel_size)
            ) for s in args.scale
        ])
        

    def forward(self, x, con=False):
        # 预处理：减去均值
        x = self.sub_mean(x)
        # 通过头部网络提取特征
        x = self.head[self.scale_idx](x)

        # 通过主体Transformer网络处理
        if not con:
            res = self.body(x, self.scale_idx)
        else:
            # 如果需要中间特征，返回中间结果
            res, x_con = self.body(x, self.scale_idx, con)
        
        # 残差连接
        res += x
        
        # 通过尾部网络上采样并输出
        x = self.tail[self.scale_idx](res)
        # 后处理：加上均值
        x = self.add_mean(x)
        
        # 根据是否需要中间特征返回不同结果
        if not con:
            return x
        else:
            return x, x_con

    # 设置当前使用的缩放因子索引
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        
# 定义视觉Transformer类
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim,          # 图像维度
        patch_dim,        # 补丁维度
        num_channels,     # 通道数
        embedding_dim,    # 嵌入维度
        num_heads,        # 注意力头数
        num_layers,       # Transformer层数
        hidden_dim,       # 隐藏层维度
        num_queries,      # 查询数量
        positional_encoding_type="learned",  # 位置编码类型
        dropout_rate=0,   # dropout率
        no_norm=False,    # 是否不使用归一化
        mlp=False,        # 是否使用MLP
        pos_every=False,  # 是否每层都使用位置编码
        no_pos=False,     # 是否不使用位置编码
        nbits=4           # 量化位数
    ):
        super(VisionTransformer, self).__init__()

        # 检查嵌入维度是否能被注意力头数整除
        assert embedding_dim % num_heads == 0
        # 检查图像维度是否能被补丁维度整除
        assert img_dim % patch_dim == 0
        
        self.no_norm = no_norm  # 是否不使用归一化
        self.mlp = mlp  # 是否使用MLP
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.patch_dim = patch_dim  # 补丁维度
        self.num_channels = num_channels  # 通道数
        
        self.img_dim = img_dim  # 图像维度
        self.pos_every = pos_every  # 是否每层都使用位置编码
        # 计算补丁数量（图像尺寸/补丁尺寸的平方）
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches  # 序列长度等于补丁数量
        # 计算展平后的补丁维度（补丁宽×补丁高×通道数）
        self.flatten_dim = patch_dim * patch_dim * num_channels
        
        self.out_dim = patch_dim * patch_dim * num_channels  # 输出维度
        
        self.no_pos = no_pos  # 是否不使用位置编码
        
        if self.mlp==False:
            # 线性编码层（量化版本）：将展平的补丁映射到嵌入维度
            self.linear_encoding = LinearLSQ(self.flatten_dim, embedding_dim, nbits_w=nbits)
            # MLP头部：将Transformer输出映射回补丁维度
            self.mlp_head = nn.Sequential(
                LinearLSQ(embedding_dim, hidden_dim, nbits_w=nbits),  # 线性层（量化）
                nn.Dropout(dropout_rate),  # dropout层
                nn.ReLU(),  # 激活函数
                LinearLSQ(hidden_dim, self.out_dim, nbits_w=nbits),  # 线性层（量化）
                nn.Dropout(dropout_rate)  # dropout层
            )

            # 查询嵌入：用于解码器的查询向量
            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        # 创建Transformer编码器层
        encoder_layer = TransformerEncoderLayer(
            embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm, nbits=nbits
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers)  # 编码器
        
        # 创建Transformer解码器层
        decoder_layer = TransformerDecoderLayer(
            embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm, nbits=nbits
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers)  # 解码器
        
        # 如果使用位置编码，初始化学习的位置编码
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                )
            
        self.dropout_layer1 = nn.Dropout(dropout_rate)  # dropout层
        
        # 如果不使用归一化，初始化线性层权重
        if no_norm:
            for m in self.modules():
                if isinstance(m, LinearLSQ):
                    nn.init.normal_(m.weight, std = 1/m.weight.size(1))

    def forward(self, x, query_idx, con=False):
        # 将图像分割为补丁：使用unfold操作提取补丁并调整维度
        x = torch.nn.functional.unfold(
            x, self.patch_dim, stride=self.patch_dim
        ).transpose(1, 2).transpose(0, 1).contiguous()
               
        if self.mlp==False:
            # 线性编码并添加残差连接
            x = self.dropout_layer1(self.linear_encoding(x, task=query_idx)) + x

            # 获取查询嵌入并调整形状
            query_embed = self.query_embed.weight[query_idx].view(
                -1, 1, self.embedding_dim
            ).repeat(1, x.size(1), 1)
        else:
            query_embed = None  # 不使用MLP时不需要查询嵌入

        # 获取位置编码（如果使用）
        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)

        # 根据不同配置处理编码器和解码器
        if self.pos_every:
            # 每层都使用位置编码
            x = self.encoder(x, task=query_idx, pos=pos)
            x = self.decoder(x, x, task=query_idx, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            # 不使用位置编码
            x = self.encoder(x, task=query_idx)
            x = self.decoder(x, x, task=query_idx, query_pos=query_embed)
        else:
            # 仅在输入时添加位置编码
            x = self.encoder(x+pos, task=query_idx)
            x = self.decoder(x, x, task=query_idx, query_pos=query_embed)
        
        # 通过MLP头部处理并添加残差连接
        if self.mlp==False:
            res = x  # 残差连接
            for layer in self.mlp_head:
                if isinstance(layer, LinearLSQ):
                    x = layer(x, task=query_idx)  # 量化线性层
                else:
                    x = layer(x)  # 其他层（dropout、激活函数）
            x = x + res  # 残差连接
        
        # 调整维度以便折叠回图像
        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)
        
        # 如果需要中间特征，返回折叠前的特征
        if con:
            con_x = x
            # 将补丁折叠回图像
            x = torch.nn.functional.fold(
                x.transpose(1, 2).contiguous(), 
                int(self.img_dim), 
                self.patch_dim, 
                stride=self.patch_dim
            )
            return x, con_x
        
        # 将补丁折叠回图像
        x = torch.nn.functional.fold(
            x.transpose(1, 2).contiguous(), 
            int(self.img_dim), 
            self.patch_dim, 
            stride=self.patch_dim
        )
        
        return x

# 定义学习的位置编码类
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        # 位置嵌入：为每个位置学习一个嵌入向量
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length  # 序列长度

        # 注册位置ID缓冲区：0到seq_length-1的位置索引
        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        # 如果没有提供位置ID，使用默认的
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        # 获取位置嵌入
        position_embeddings = self.pe(position_ids)
        return position_embeddings
    
# 定义Transformer编码器类
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # 复制多个编码器层
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers  # 层数

    def forward(self, src, task, pos = None):
        output = src  # 初始输入

        # 逐层处理
        for layer in self.layers:
            output = layer(output, task, pos=pos)

        return output
    
# 定义Transformer编码器层类
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu", nbits=4):
        super().__init__()
               
        # 自注意力层（量化版本）
        self.self_attn = QuantMultiheadAttention(
            d_model, nhead, n_bit=nbits, dropout=dropout, bias=False, encoder=False
        )
        
        # 前馈网络
        self.linear1 = LinearLSQ(d_model, dim_feedforward, nbits_w=nbits)  # 线性层1（量化）
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.linear2 = LinearLSQ(dim_feedforward, d_model, nbits_w=nbits)  # 线性层2（量化）
        
        # 归一化层（如果使用）
        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)  # 注意力后的dropout
        self.dropout2 = nn.Dropout(dropout)  # 前馈网络后的dropout

        self.activation = _get_activation_fn(activation)  # 获取激活函数
        
        # 初始化自注意力层权重
        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    # 为张量添加位置编码
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, task, pos = None):
        # 自注意力部分
        src2 = self.norm1(src)  # 归一化
        q = k = self.with_pos_embed(src2, pos)  # 查询和键添加位置编码
        # 通过自注意力层
        src2 = self.self_attn(q, k, src2, task=task)
        # 残差连接和dropout
        src = src + self.dropout1(src2[0])
        
        # 前馈网络部分
        src2 = self.norm2(src)  # 归一化
        # 线性层1 -> 激活函数 -> dropout -> 线性层2
        src2 = self.linear2(
            self.dropout(self.activation(self.linear1(src2, task=task))), 
            task=task
        )
        # 残差连接和dropout
        src = src + self.dropout2(src2)
        return src

    
# 定义Transformer解码器类
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        # 复制多个解码器层
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers  # 层数

    def forward(self, tgt, memory, task, pos = None, query_pos = None):
        output = tgt  # 初始目标输入
        
        # 逐层处理
        for layer in self.layers:
            output = layer(output, memory, task, pos=pos, query_pos=query_pos)

        return output

    
# 定义Transformer解码器层类
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu", nbits=4):
        super().__init__()
        
        # 自注意力层（量化版本）：用于解码器内部的注意力
        self.self_attn = QuantMultiheadAttention(
            d_model, nhead, n_bit=nbits, dropout=dropout, bias=False, encoder=False
        )
        
        # 多头注意力层（量化版本）：用于解码器-编码器注意力
        self.multihead_attn = QuantMultiheadAttention(
            d_model, nhead, n_bit=nbits, dropout=dropout, bias=False, encoder=True
        )
        
        # 前馈网络
        self.linear1 = LinearLSQ(d_model, dim_feedforward, nbits_w=nbits)  # 线性层1（量化）
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.linear2 = LinearLSQ(dim_feedforward, d_model, nbits_w=nbits)  # 线性层2（量化）

        # 归一化层（如果使用）
        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)  # 自注意力后的dropout
        self.dropout2 = nn.Dropout(dropout)  # 解码器-编码器注意力后的dropout
        self.dropout3 = nn.Dropout(dropout)  # 前馈网络后的dropout

        self.activation = _get_activation_fn(activation)  # 获取激活函数

    # 为张量添加位置编码
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, task, pos = None, query_pos = None):
        # 自注意力部分
        tgt2 = self.norm1(tgt)  # 归一化
        q = k = self.with_pos_embed(tgt2, query_pos)  # 查询和键添加查询位置编码
        # 通过自注意力层
        tgt2 = self.self_attn(q, k, value=tgt2, task=task)[0]
        # 残差连接和dropout
        tgt = tgt + self.dropout1(tgt2)
        
        # 解码器-编码器注意力部分
        tgt2 = self.norm2(tgt)  # 归一化
        query = self.with_pos_embed(tgt2, query_pos)  # 查询添加查询位置编码
        key = self.with_pos_embed(memory, pos)  # 键添加位置编码
        value = memory  # 值
        # 通过多头注意力层
        tgt2 = self.multihead_attn(query, key, value=value, task=task)[0]
        # 残差连接和dropout
        tgt = tgt + self.dropout2(tgt2)
        
        # 前馈网络部分
        tgt2 = self.norm3(tgt)  # 归一化
        # 线性层1 -> 激活函数 -> dropout -> 线性层2
        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(tgt2, task=task))), 
            task=task
        )
        # 残差连接和dropout
        tgt = tgt + self.dropout3(tgt2)
        return tgt


# 复制模块N次的辅助函数
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# 获取激活函数的辅助函数
def _get_activation_fn(activation):
    """根据字符串返回对应的激活函数"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"激活函数应该是relu/gelu，而不是{activation}。")







# from model import common_qipt
# from model import common
# from model import Quant
# import math
# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor
# from einops import rearrange
# import copy

# from .lsq_plus import *
# from ._quan_base_plus import *

# from .attention_layer import *


# def make_model(args, parent=False):
#     return ipt(args)

# class ipt(nn.Module):
#     def __init__(self, args, conv=Conv2dLSQ):
#         super(ipt, self).__init__()
        
#         self.scale_idx = 0
        
#         self.args = args
#         nbits = 4
        
#         n_feats = args.n_feats
#         kernel_size = 3 
#         act = nn.ReLU(True)

#         self.sub_mean = common_qipt.MeanShift(args.rgb_range)
#         self.add_mean = common_qipt.MeanShift(args.rgb_range, sign=1)
#         # import pdb; pdb.set_trace()
#         self.head = nn.ModuleList([
#             nn.Sequential(
#                 common_qipt.default_conv(args.n_colors, n_feats, kernel_size),
#                 common_qipt.ResBlock(conv, n_feats, 5, act=act, nbits=nbits),
#                 common_qipt.ResBlock(conv, n_feats, 5, act=act, nbits=nbits)
#             ) for _ in args.scale
#         ])

#         self.body = VisionTransformer(img_dim=args.patch_size, patch_dim=args.patch_dim, num_channels=n_feats, embedding_dim=n_feats*args.patch_dim*args.patch_dim, num_heads=args.num_heads, num_layers=args.num_layers, hidden_dim=n_feats*args.patch_dim*args.patch_dim*4, num_queries = args.num_queries, dropout_rate=args.dropout_rate, mlp=args.no_mlp ,pos_every=args.pos_every,no_pos=args.no_pos,no_norm=args.no_norm, nbits=nbits)
#         # import pdb; pdb.set_trace()
#         self.tail = nn.ModuleList([
#             nn.Sequential(
#                 common_qipt.Upsampler(common_qipt.default_conv, s, n_feats, act=False),
#                 #common_qipt.Upsampler(conv, s, n_feats, act=False),
#                 common_qipt.default_conv(n_feats, args.n_colors, kernel_size)
#             ) for s in args.scale
#         ])
        

#     def forward(self, x, con=False):
#         # import pdb; pdb.set_trace()
#         x = self.sub_mean(x)
#         x = self.head[self.scale_idx](x)

#         if not con:
#             res = self.body(x, self.scale_idx)
#         else:
#             res, x_con = self.body(x, self.scale_idx, con)
        
#         # res = self.body(x, self.scale_idx, con)
#         res += x
        
#         # import pdb; pdb.set_trace()
#         x = self.tail[self.scale_idx](res)
#         x = self.add_mean(x)
        
#         if not con:
#             return x
#         else:
#             return x, x_con
#         # return x 

#     def set_scale(self, scale_idx):
#         self.scale_idx = scale_idx
        
# class VisionTransformer(nn.Module):
#     def __init__(
#         self,
#         img_dim,
#         patch_dim,
#         num_channels,
#         embedding_dim,
#         num_heads,
#         num_layers,
#         hidden_dim,
#         num_queries,
#         positional_encoding_type="learned",
#         dropout_rate=0,
#         no_norm=False,
#         mlp=False,
#         pos_every=False,
#         no_pos = False,
#         nbits=4
#     ):
#         super(VisionTransformer, self).__init__()

#         assert embedding_dim % num_heads == 0
#         assert img_dim % patch_dim == 0
#         self.no_norm = no_norm
#         self.mlp = mlp
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.patch_dim = patch_dim
#         self.num_channels = num_channels
        
#         self.img_dim = img_dim
#         self.pos_every = pos_every
#         self.num_patches = int((img_dim // patch_dim) ** 2)
#         self.seq_length = self.num_patches
#         self.flatten_dim = patch_dim * patch_dim * num_channels
        
#         self.out_dim = patch_dim * patch_dim * num_channels
        
#         self.no_pos = no_pos
        
#         if self.mlp==False:
#             self.linear_encoding = LinearLSQ(self.flatten_dim, embedding_dim, nbits_w=nbits)
#             self.mlp_head = nn.Sequential(
#                 LinearLSQ(embedding_dim, hidden_dim, nbits_w=nbits),
#                 nn.Dropout(dropout_rate),
#                 nn.ReLU(),
#                 LinearLSQ(hidden_dim, self.out_dim, nbits_w=nbits),
#                 nn.Dropout(dropout_rate)
#             )

#             # import pdb; pdb.set_trace()
#             self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

#         encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm, nbits=nbits)
#         self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
#         decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm, nbits=nbits)
#         self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
#         if not self.no_pos:
#             self.position_encoding = LearnedPositionalEncoding(
#                     self.seq_length, self.embedding_dim, self.seq_length
#                 )
            
#         self.dropout_layer1 = nn.Dropout(dropout_rate)
        
#         if no_norm:
#             for m in self.modules():
#                 if isinstance(m, LinearLSQ):
#                     nn.init.normal_(m.weight, std = 1/m.weight.size(1))

#     def forward(self, x, query_idx, con=False):

#         x = torch.nn.functional.unfold(x,self.patch_dim,stride=self.patch_dim).transpose(1,2).transpose(0,1).contiguous()
               
#         if self.mlp==False:
#             x = self.dropout_layer1(self.linear_encoding(x, task=query_idx)) + x

#             query_embed = self.query_embed.weight[query_idx].view(-1,1,self.embedding_dim).repeat(1,x.size(1), 1)
#             # query_embed = self.query_embed.weight[0].view(-1,1,self.embedding_dim).repeat(1,x.size(1), 1)
#         else:
#             query_embed = None

        
#         if not self.no_pos:
#             pos = self.position_encoding(x).transpose(0,1)

#         if self.pos_every:
#             x = self.encoder(x, task=query_idx, pos=pos)
#             x = self.decoder(x, x, task=query_idx, pos=pos, query_pos=query_embed)
#         elif self.no_pos:
#             x = self.encoder(x, task=query_idx)
#             x = self.decoder(x, x, task=query_idx, query_pos=query_embed)
#         else:
#             x = self.encoder(x+pos, task=query_idx)
#             x = self.decoder(x, x, task=query_idx, query_pos=query_embed)
        
        
#         if self.mlp==False:
#             res = x
#             for layer in self.mlp_head:
#                 if isinstance(layer, LinearLSQ):
#                     x = layer(x, task=query_idx)
#                 else:
#                     x = layer(x)
#             x = x + res
        
#         x = x.transpose(0,1).contiguous().view(x.size(1), -1, self.flatten_dim)
        
#         # import pdb; pdb.set_trace()
#         if con:
#             con_x = x
#             x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
#             return x, con_x
        
#         x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
        
#         return x

# class LearnedPositionalEncoding(nn.Module):
#     def __init__(self, max_position_embeddings, embedding_dim, seq_length):
#         super(LearnedPositionalEncoding, self).__init__()
#         self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
#         self.seq_length = seq_length

#         self.register_buffer(
#             "position_ids", torch.arange(self.seq_length).expand((1, -1))
#         )

#     def forward(self, x, position_ids=None):
#         if position_ids is None:
#             position_ids = self.position_ids[:, : self.seq_length]

#         position_embeddings = self.pe(position_ids)
#         return position_embeddings
    
# class TransformerEncoder(nn.Module):

#     def __init__(self, encoder_layer, num_layers):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers

#     def forward(self, src, task, pos = None):
#         output = src

#         for layer in self.layers:
#             output = layer(output, task, pos=pos)

#         return output
    
# class TransformerEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
#                  activation="relu", nbits=4):
#         super().__init__()
               
#         self.self_attn = QuantMultiheadAttention(d_model, nhead, n_bit=nbits, dropout=dropout, bias=False, encoder=False)
#         # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
#         # self.self_attn.out_proj = Quant.LinearQ(d_model, d_model, bias=False, nbits_w=4)
#         # self.q_act = Quant.ActQ(nbits_a=4, in_features=d_model)
#         # self.k_act = Quant.ActQ(nbits_a=4, in_features=d_model)
#         # self.v_act = Quant.ActQ(nbits_a=4, in_features=d_model)
#         # Implementation of Feedforward model
#         self.linear1 = LinearLSQ(d_model, dim_feedforward, nbits_w=nbits)
#         # self.linear1 = Quant.LinearQ(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = LinearLSQ(dim_feedforward, d_model, nbits_w=nbits)
#         # self.linear2 = Quant.LinearQ(dim_feedforward, d_model)
        
#         self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
        
#         nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

#     def with_pos_embed(self, tensor, pos):
#         return tensor if pos is None else tensor + pos
    
#     def forward(self, src, task, pos = None):
#         # import pdb; pdb.set_trace()
#         src2 = self.norm1(src)
#         q = k = self.with_pos_embed(src2, pos)
#         # q = self.q_act(q)
#         # k = self.k_act(k)
#         # src2 = self.v_act(src2)
#         src2 = self.self_attn(q, k, src2, task=task)
#         src = src + self.dropout1(src2[0])
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2, task=task))), task=task)
#         src = src + self.dropout2(src2)
#         return src

    
# class TransformerDecoder(nn.Module):

#     def __init__(self, decoder_layer, num_layers):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers

#     def forward(self, tgt, memory, task, pos = None, query_pos = None):
#         output = tgt
        
#         for layer in self.layers:
#             output = layer(output, memory, task, pos=pos, query_pos=query_pos)

#         return output

    
# class TransformerDecoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
#                  activation="relu", nbits=4):
#         super().__init__()
        
   
#         # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
#         # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
#         self.self_attn = QuantMultiheadAttention(d_model, nhead, n_bit=nbits, dropout=dropout, bias=False, encoder=False)
#         #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
#         #self.self_attn.out_proj = Quant.LinearQ(d_model, d_model, bias=False, nbits_w=4)
#         #self.q_act = Quant.ActQ(nbits_a=4, in_features=d_model)
#         #self.k_act = Quant.ActQ(nbits_a=4, in_features=d_model)
#         #self.v_act = Quant.ActQ(nbits_a=4, in_features=d_model)
#         self.multihead_attn = QuantMultiheadAttention(d_model, nhead, n_bit=nbits, dropout=dropout, bias=False, encoder=True)
#         #self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
#         #self.multihead_attn.out_proj = Quant.LinearQ(d_model, d_model, bias=False, nbits_w=4)
#         #self.mq_act = Quant.ActQ(nbits_a=4, in_features=d_model)
#         #self.mk_act = Quant.ActQ(nbits_a=4, in_features=d_model)
#         #self.mv_act = Quant.ActQ(nbits_a=4, in_features=d_model)
#         # Implementation of Feedforward model
#         self.linear1 = LinearLSQ(d_model, dim_feedforward, nbits_w=nbits)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = LinearLSQ(dim_feedforward, d_model, nbits_w=nbits)
#         #self.linear1 = Quant.LinearQ(d_model, dim_feedforward)
#         #self.dropout = nn.Dropout(dropout)
#         #self.linear2 = Quant.LinearQ(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)

#     def with_pos_embed(self, tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward(self, tgt, memory, task, pos = None, query_pos = None):
#         # import pdb; pdb.set_trace()
#         tgt2 = self.norm1(tgt)
#         q = k = self.with_pos_embed(tgt2, query_pos)
#         #q = self.q_act(q)
#         #k = self.k_act(k)
#         #tgt2 = self.v_act(tgt2)
#         tgt2 = self.self_attn(q, k, value=tgt2, task=task)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2 = self.norm2(tgt)
#         query = self.with_pos_embed(tgt2, query_pos)
#         key = self.with_pos_embed(memory, pos)
#         value = memory
#         #query = self.mq_act(query)
#         #key = self.mk_act(key)
#         #value = self.mv_act(value)
#         tgt2 = self.multihead_attn(query,
#                                    key,
#                                    value=value, task=task)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt2 = self.norm3(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2, task=task))), task=task)
#         tgt = tgt + self.dropout3(tgt2)
#         return tgt


# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")





