import os
from importlib import import_module  # 用于动态导入模块

import torch
import torch.nn as nn
import torch.nn.parallel as P  # 用于多GPU并行计算
import torch.utils.model_zoo  # 用于下载预训练模型

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args                                                    # 保留传入的参数对象
        self.scale = args.scale                                             # 保留缩放比例列表（用于超分辨率等任务）
        self.patch_size = args.patch_size                                   # 保留处理块（补丁）大小参数
        self.idx_scale = 0                                                  # 当前使用的scale索引（多尺度任务时切换）
        self.input_large = (args.model == 'VDSR')                           # 特定模型(VDSR)的特殊标识，用于输入处理
        self.self_ensemble = args.self_ensemble                             # 是否启用自我集成（测试时的数据增强）
        self.precision = args.precision                                     # 数值精度（如'single'单精度或'half'半精度）
        self.cpu = args.cpu                                                 # 是否强制使用CPU
        self.device = torch.device('cpu' if args.cpu else 'cuda')           # 选择计算设备（CPU或GPU）
        self.n_GPUs = args.n_GPUs                                           # 使用的GPU数量（多GPU训练）
        self.save_models = args.save_models                                 # 是否按epoch保存多个模型文件
        
        # 动态导入模型模块：根据模型名称从model包中导入对应的模块
        module = import_module('model.' + args.model.lower())
        # 创建模型实例并移动到指定设备
        self.model = module.make_model(args).to(self.device)
        # 如果需要半精度计算，将模型参数转为float16
        if args.precision == 'half':
            self.model.half()
        
        # 以下代码用于加载检查点和日志打印，目前被注释
        # self.load(
        #     ckp.get_path('model'),
        #     resume=args.resume,
        #     cpu=args.cpu
        # )
        # print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale, con=False):
        # 设置当前使用的scale索引
        self.idx_scale = idx_scale
        
        # 如果模型有set_scale方法，调用它设置当前scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        # 训练模式
        if self.training:
            # 单GPU或CPU训练直接调用模型
            return self.model(x, con)
        # 测试模式
        else:
            # 使用分块推理函数（处理大图像时避免内存溢出）
            forward_function = self.forward_chop

            # 如果启用自我集成（测试增强）
            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        # 定义模型保存路径列表
        save_dirs = [os.path.join(apath, 'model_latest.pt')]  # 最新模型

        # 如果是最佳模型，添加最佳模型保存路径
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        # 如果需要保存每个epoch的模型
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        # 保存模型参数到所有指定路径
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None  # 用于存储加载的模型参数
        kwargs = {}
        # 如果指定使用CPU，设置映射位置
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        # 加载最新模型
        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        # 加载预训练模型
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)  # 创建模型保存目录
                # 从URL下载预训练模型
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,** kwargs
                )
        # 加载指定epoch的模型
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        # 如果成功加载模型参数，加载到当前模型
        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward_x8(self, *args, forward_function=None):
        """实现8种变换的自我集成（数据增强）"""
        def _transform(v, op):
            # 如果不是单精度，先转为float32处理
            if self.precision != 'single': v = v.float()

            # 转为numpy数组进行变换
            v2np = v.data.cpu().numpy()
            if op == 'v':  # 垂直翻转
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':  # 水平翻转
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':  # 转置
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            # 转回张量并移动到指定设备
            ret = torch.Tensor(tfnp).to(self.device)
            # 如果需要半精度，转回float16
            if self.precision == 'half': ret = ret.half()

            return ret

        # 生成所有8种变换的输入
        list_x = []
        for a in args:
            x = [a]  # 原始输入
            # 对原始输入进行v, h, t三种变换，生成8种组合
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])
            list_x.append(x)

        # 对所有变换后的输入进行推理
        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            # 初始化结果列表
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        # 对输出进行逆变换，恢复原始方向
        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:  # 转置的逆变换
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:  # 水平翻转的逆变换
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:  # 垂直翻转的逆变换
                    _list_y[i] = _transform(_list_y[i], 'v')

        # 对所有变换的结果取平均
        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y
    
    def forward_chop(self, x, shave=12):
        """分块推理：将大图像分成小块处理，避免内存溢出"""
        x.cpu()  # 将输入移到CPU（准备分块）
        batchsize = self.args.crop_batch_size  # 分块处理的批次大小
        h, w = x.size()[-2:]  # 获取输入图像的高度和宽度
        padsize = int(self.patch_size)  # 块大小
        shave = int(self.patch_size/2)  # 块之间的重叠大小（用于消除拼接边界）

        scale = self.scale[self.idx_scale]  # 当前缩放比例

        # 计算需要裁剪的大小，确保图像尺寸能被块大小整除
        h_cut = (h-padsize)%(int(shave/2))
        w_cut = (w-padsize)%(int(shave/2))

        # 将图像分块（unfold操作）
        x_unfold = torch.nn.functional.unfold(
            x, padsize, stride=int(shave/2)
        ).transpose(0,2).contiguous()

        # 处理右下角边缘块
        x_hw_cut = x[...,(h-padsize):,(w-padsize):]
        y_hw_cut = self.model.forward(x_hw_cut.cuda()).cpu()

        # 处理底部和右侧边缘块
        x_h_cut = x[...,(h-padsize):,:]
        x_w_cut = x[...,:,(w-padsize):]
        y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        
        # 处理顶部和左侧边缘块
        x_h_top = x[...,:padsize,:]
        x_w_top = x[...,:,:padsize]
        y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        # 处理中间块
        x_unfold = x_unfold.view(x_unfold.size(0),-1,padsize,padsize)
        y_unfold = []

        # 计算分块批次数量
        x_range = x_unfold.size(0)//batchsize + (x_unfold.size(0)%batchsize !=0)
        x_unfold.cuda()  # 将分块移到GPU
        # 分批处理中间块
        for i in range(x_range):
            y_unfold.append(self.model(x_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())
        y_unfold = torch.cat(y_unfold,dim=0)  # 拼接所有批次结果

        # 将分块结果折叠回完整图像
        y = torch.nn.functional.fold(
            y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),
            ((h-h_cut)*scale,(w-w_cut)*scale), 
            padsize*scale, 
            stride=int(shave/2*scale)
        )
        
        # 替换边缘块结果（顶部和左侧）
        y[...,:padsize*scale,:] = y_h_top
        y[...,:,:padsize*scale] = y_w_top

        # 处理中间块重叠区域的平均
        y_unfold = y_unfold[
            ...,
            int(shave/2*scale):padsize*scale-int(shave/2*scale),
            int(shave/2*scale):padsize*scale-int(shave/2*scale)
        ].contiguous()
        y_inter = torch.nn.functional.fold(
            y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),
            ((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), 
            padsize*scale-shave*scale, 
            stride=int(shave/2*scale)
        )
        
        # 计算重叠区域的平均权重
        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(
                y_ones, 
                padsize*scale-shave*scale, 
                stride=int(shave/2*scale)
            ),
            ((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), 
            padsize*scale-shave*scale, 
            stride=int(shave/2*scale)
        )
        
        y_inter = y_inter/divisor  # 重叠区域取平均

        # 替换中间区域结果
        y[
            ...,
            int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),
            int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)
        ] = y_inter

        # 拼接底部和右侧边缘块
        y = torch.cat([
            y[...,:y.size(2)-int((padsize-h_cut)/2*scale),:],
            y_h_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]
        ],dim=2)
        y_w_cat = torch.cat([
            y_w_cut[...,:y_w_cut.size(2)-int((padsize-h_cut)/2*scale),:],
            y_hw_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]
        ],dim=2)
        y = torch.cat([
            y[...,:,:y.size(3)-int((padsize-w_cut)/2*scale)],
            y_w_cat[...,:,int((padsize-w_cut)/2*scale+0.5):]
        ],dim=3)
        
        return y.cuda()  # 将最终结果移回GPU
    
    def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        """处理高度方向的边缘块（顶部或底部）"""
        
        # 将输入分块
        x_h_cut_unfold = torch.nn.functional.unfold(
            x_h_cut, padsize, stride=int(shave/2)
        ).transpose(0,2).contiguous()
        
        # 调整分块形状
        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0),-1,padsize,padsize)
        # 计算分块批次数量
        x_range = x_h_cut_unfold.size(0)//batchsize + (x_h_cut_unfold.size(0)%batchsize !=0)
        y_h_cut_unfold=[]
        x_h_cut_unfold.cuda()  # 将分块移到GPU
        # 分批处理
        for i in range(x_range):
            y_h_cut_unfold.append(
                self.model(x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...]).cpu()
            )
        y_h_cut_unfold = torch.cat(y_h_cut_unfold,dim=0)  # 拼接结果
        
        # 将分块结果折叠回图像
        y_h_cut = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),
            (padsize*scale,(w-w_cut)*scale), 
            padsize*scale, 
            stride=int(shave/2*scale)
        )
        
        # 处理重叠区域的平均
        y_h_cut_unfold = y_h_cut_unfold[
            ...,
            :,
            int(shave/2*scale):padsize*scale-int(shave/2*scale)
        ].contiguous()
        y_h_cut_inter = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),
            (padsize*scale,(w-w_cut-shave)*scale), 
            (padsize*scale,padsize*scale-shave*scale), 
            stride=int(shave/2*scale)
        )
        
        # 计算重叠区域的平均权重
        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(
                y_ones,
                (padsize*scale,padsize*scale-shave*scale), 
                stride=int(shave/2*scale)
            ),
            (padsize*scale,(w-w_cut-shave)*scale), 
            (padsize*scale,padsize*scale-shave*scale), 
            stride=int(shave/2*scale)
        )
        y_h_cut_inter = y_h_cut_inter/divisor  # 重叠区域取平均
        
        # 替换重叠区域结果
        y_h_cut[
            ...,
            :,
            int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)
        ] = y_h_cut_inter
        return y_h_cut
        
    def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        """处理宽度方向的边缘块（左侧或右侧）"""
        
        # 将输入分块
        x_w_cut_unfold = torch.nn.functional.unfold(
            x_w_cut, padsize, stride=int(shave/2)
        ).transpose(0,2).contiguous()
        
        # 调整分块形状
        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0),-1,padsize,padsize)
        # 计算分块批次数量
        x_range = x_w_cut_unfold.size(0)//batchsize + (x_w_cut_unfold.size(0)%batchsize !=0)
        y_w_cut_unfold=[]
        x_w_cut_unfold.cuda()  # 将分块移到GPU
        # 分批处理
        for i in range(x_range):
            y_w_cut_unfold.append(
                self.model(x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...]).cpu()
            )
        y_w_cut_unfold = torch.cat(y_w_cut_unfold,dim=0)  # 拼接结果
        
        # 将分块结果折叠回图像
        y_w_cut = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),
            ((h-h_cut)*scale,padsize*scale), 
            padsize*scale, 
            stride=int(shave/2*scale)
        )
        
        # 处理重叠区域的平均
        y_w_cut_unfold = y_w_cut_unfold[
            ...,
            int(shave/2*scale):padsize*scale-int(shave/2*scale),
            :
        ].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),
            ((h-h_cut-shave)*scale,padsize*scale), 
            (padsize*scale-shave*scale,padsize*scale), 
            stride=int(shave/2*scale)
        )
        
        # 计算重叠区域的平均权重
        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(
                y_ones,
                (padsize*scale-shave*scale,padsize*scale), 
                stride=int(shave/2*scale)
            ),
            ((h-h_cut-shave)*scale,padsize*scale), 
            (padsize*scale-shave*scale,padsize*scale), 
            stride=int(shave/2*scale)
        )
        y_w_cut_inter = y_w_cut_inter/divisor  # 重叠区域取平均

        # 替换重叠区域结果
        y_w_cut[
            ...,
            int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),
            :
        ] = y_w_cut_inter
        return y_w_cut





# import os
# from importlib import import_module

# import torch
# import torch.nn as nn
# import torch.nn.parallel as P
# import torch.utils.model_zoo

# class Model(nn.Module):
#     def __init__(self, args, ckp):
#         super(Model, self).__init__()
#         print('Making model...')
#         self.args = args                                                    # 保留传入参数对象。
#         self.scale = args.scale                                             # 保留缩放比例列表。              
#         self.patch_size = args.patch_size                                   # 保留处理块（补丁）大小参数。
#         self.idx_scale = 0                                                  # 当前使用的 scale 索引（多尺度时切换）。
#         self.input_large = (args.model == 'VDSR')                           # 基于模型名判断是否有“输入很大”的特殊处理（只是一个布尔标识）。
#         self.self_ensemble = args.self_ensemble                             # 是否启用自我集成（测试时增强）。
#         self.precision = args.precision                                     # 数值精度（如 'single' 或 'half'）。   
#         self.cpu = args.cpu                                                 # 是否强制使用 CPU（否则使用 GPU）。
#         self.device = torch.device('cpu' if args.cpu else 'cuda')           # 选择设备（CPU 或 GPU）。
#         self.n_GPUs = args.n_GPUs                                           # 使用的 GPU 数量（多 GPU 训练）。
#         self.save_models = args.save_models                                 # 是否按 epoch 保存多个模型文件。            
        
#         module = import_module('model.' + args.model.lower())               # 动态的导入模块，根据model的名字从model包中导入对应的模块
#         self.model = module.make_model(args).to(self.device)                # 调用模块的make_model函数，根据参数创建模型实例，并移动到指定设备
#         if args.precision == 'half':                                        # 把模型参数转为半精度（float16）。
#             self.model.half()
        
#         # 导入检查点，将模型打印到日志
#         # self.load(
#         #     ckp.get_path('model'),
#         #     resume=args.resume,
#         #     cpu=args.cpu
#         # )
        
#         # print(self.model, file=ckp.log_file)

#     def forward(self, x, idx_scale, con=False):
#         # import pdb; pdb.set_trace()
#         self.idx_scale = idx_scale
        
#         if hasattr(self.model, 'set_scale'):
#             self.model.set_scale(idx_scale)

#         if self.training:
#             # if self.n_GPUs > 1:
#             #     return P.data_parallel(self.model, x, range(self.n_GPUs))
#             # else:
#                 return self.model(x, con)
#         else:
#             forward_function = self.forward_chop

#             if self.self_ensemble:
#                 return self.forward_x8(x, forward_function=forward_function)
#             else:
#                 return forward_function(x)

#     def save(self, apath, epoch, is_best=False):
#         save_dirs = [os.path.join(apath, 'model_latest.pt')]

#         if is_best:
#             save_dirs.append(os.path.join(apath, 'model_best.pt'))
#         if self.save_models:
#             save_dirs.append(
#                 os.path.join(apath, 'model_{}.pt'.format(epoch))
#             )

#         for s in save_dirs:
#             torch.save(self.model.state_dict(), s)

#     def load(self, apath, pre_train='', resume=-1, cpu=False):
#         load_from = None
#         kwargs = {}
#         if cpu:
#             kwargs = {'map_location': lambda storage, loc: storage}

#         if resume == -1:
#             load_from = torch.load(
#                 os.path.join(apath, 'model_latest.pt'),
#                 **kwargs
#             )
#         elif resume == 0:
#             if pre_train == 'download':
#                 print('Download the model')
#                 dir_model = os.path.join('..', 'models')
#                 os.makedirs(dir_model, exist_ok=True)
#                 load_from = torch.utils.model_zoo.load_url(
#                     self.model.url,
#                     model_dir=dir_model,
#                     **kwargs
#                 )
#         else:
#             load_from = torch.load(
#                 os.path.join(apath, 'model_{}.pt'.format(resume)),
#                 **kwargs
#             )

#         if load_from:
#             self.model.load_state_dict(load_from, strict=False)

#     def forward_x8(self, *args, forward_function=None):
#         def _transform(v, op):
#             if self.precision != 'single': v = v.float()

#             v2np = v.data.cpu().numpy()
#             if op == 'v':
#                 tfnp = v2np[:, :, :, ::-1].copy()
#             elif op == 'h':
#                 tfnp = v2np[:, :, ::-1, :].copy()
#             elif op == 't':
#                 tfnp = v2np.transpose((0, 1, 3, 2)).copy()

#             ret = torch.Tensor(tfnp).to(self.device)
#             if self.precision == 'half': ret = ret.half()

#             return ret

#         list_x = []
#         for a in args:
#             x = [a]
#             for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

#             list_x.append(x)

#         list_y = []
#         for x in zip(*list_x):
#             y = forward_function(*x)
#             if not isinstance(y, list): y = [y]
#             if not list_y:
#                 list_y = [[_y] for _y in y]
#             else:
#                 for _list_y, _y in zip(list_y, y): _list_y.append(_y)

#         for _list_y in list_y:
#             for i in range(len(_list_y)):
#                 if i > 3:
#                     _list_y[i] = _transform(_list_y[i], 't')
#                 if i % 4 > 1:
#                     _list_y[i] = _transform(_list_y[i], 'h')
#                 if (i % 4) % 2 == 1:
#                     _list_y[i] = _transform(_list_y[i], 'v')

#         y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
#         if len(y) == 1: y = y[0]

#         return y
    
#     def forward_chop(self, x, shave=12):
#         x.cpu()
#         batchsize = self.args.crop_batch_size
#         h, w = x.size()[-2:]
#         padsize = int(self.patch_size)
#         shave = int(self.patch_size/2)

#         scale = self.scale[self.idx_scale]

#         h_cut = (h-padsize)%(int(shave/2))
#         w_cut = (w-padsize)%(int(shave/2))

#         x_unfold = torch.nn.functional.unfold(x, padsize, stride=int(shave/2)).transpose(0,2).contiguous()

#         x_hw_cut = x[...,(h-padsize):,(w-padsize):]
#         y_hw_cut = self.model.forward(x_hw_cut.cuda()).cpu()

#         x_h_cut = x[...,(h-padsize):,:]
#         x_w_cut = x[...,:,(w-padsize):]
#         y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
#         y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        
#         x_h_top = x[...,:padsize,:]
#         x_w_top = x[...,:,:padsize]
#         y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
#         y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

#         x_unfold = x_unfold.view(x_unfold.size(0),-1,padsize,padsize)
#         y_unfold = []

#         x_range = x_unfold.size(0)//batchsize + (x_unfold.size(0)%batchsize !=0)
#         x_unfold.cuda()
#         for i in range(x_range):
#             # y_unfold.append(P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
#             y_unfold.append(self.model(x_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())
#         y_unfold = torch.cat(y_unfold,dim=0)

#         y = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut)*scale,(w-w_cut)*scale), padsize*scale, stride=int(shave/2*scale))
        
#         y[...,:padsize*scale,:] = y_h_top
#         y[...,:,:padsize*scale] = y_w_top

#         y_unfold = y_unfold[...,int(shave/2*scale):padsize*scale-int(shave/2*scale),int(shave/2*scale):padsize*scale-int(shave/2*scale)].contiguous()
#         y_inter = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), padsize*scale-shave*scale, stride=int(shave/2*scale))
        
#         y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
#         divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones, padsize*scale-shave*scale, stride=int(shave/2*scale)),((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), padsize*scale-shave*scale, stride=int(shave/2*scale))
        
#         y_inter = y_inter/divisor

#         y[...,int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)] = y_inter

#         y = torch.cat([y[...,:y.size(2)-int((padsize-h_cut)/2*scale),:],y_h_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2)
#         y_w_cat = torch.cat([y_w_cut[...,:y_w_cut.size(2)-int((padsize-h_cut)/2*scale),:],y_hw_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2)
#         y = torch.cat([y[...,:,:y.size(3)-int((padsize-w_cut)/2*scale)],y_w_cat[...,:,int((padsize-w_cut)/2*scale+0.5):]],dim=3)
#         return y.cuda()
    
#     def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        
#         x_h_cut_unfold = torch.nn.functional.unfold(x_h_cut, padsize, stride=int(shave/2)).transpose(0,2).contiguous()
        
#         x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0),-1,padsize,padsize)
#         x_range = x_h_cut_unfold.size(0)//batchsize + (x_h_cut_unfold.size(0)%batchsize !=0)
#         y_h_cut_unfold=[]
#         x_h_cut_unfold.cuda()
#         for i in range(x_range):
#             # y_h_cut_unfold.append(P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
#             y_h_cut_unfold.append(self.model(x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())
#         y_h_cut_unfold = torch.cat(y_h_cut_unfold,dim=0)
        
#         y_h_cut = torch.nn.functional.fold(y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),(padsize*scale,(w-w_cut)*scale), padsize*scale, stride=int(shave/2*scale))
#         y_h_cut_unfold = y_h_cut_unfold[...,:,int(shave/2*scale):padsize*scale-int(shave/2*scale)].contiguous()
#         y_h_cut_inter = torch.nn.functional.fold(y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),(padsize*scale,(w-w_cut-shave)*scale), (padsize*scale,padsize*scale-shave*scale), stride=int(shave/2*scale))
        
#         y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
#         divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones ,(padsize*scale,padsize*scale-shave*scale), stride=int(shave/2*scale)),(padsize*scale,(w-w_cut-shave)*scale), (padsize*scale,padsize*scale-shave*scale), stride=int(shave/2*scale)) 
#         y_h_cut_inter = y_h_cut_inter/divisor
        
#         y_h_cut[...,:,int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)] = y_h_cut_inter
#         return y_h_cut
        
#     def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        
#         x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave/2)).transpose(0,2).contiguous()
        
#         x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0),-1,padsize,padsize)
#         x_range = x_w_cut_unfold.size(0)//batchsize + (x_w_cut_unfold.size(0)%batchsize !=0)
#         y_w_cut_unfold=[]
#         x_w_cut_unfold.cuda()
#         for i in range(x_range):
#             # y_w_cut_unfold.append(P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
#             y_w_cut_unfold.append(self.model(x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())
#         y_w_cut_unfold = torch.cat(y_w_cut_unfold,dim=0)
        
#         y_w_cut = torch.nn.functional.fold(y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut)*scale,padsize*scale), padsize*scale, stride=int(shave/2*scale))
#         y_w_cut_unfold = y_w_cut_unfold[...,int(shave/2*scale):padsize*scale-int(shave/2*scale),:].contiguous()
#         y_w_cut_inter = torch.nn.functional.fold(y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut-shave)*scale,padsize*scale), (padsize*scale-shave*scale,padsize*scale), stride=int(shave/2*scale))
        
#         y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
#         divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones ,(padsize*scale-shave*scale,padsize*scale), stride=int(shave/2*scale)),((h-h_cut-shave)*scale,padsize*scale), (padsize*scale-shave*scale,padsize*scale), stride=int(shave/2*scale))
#         y_w_cut_inter = y_w_cut_inter/divisor

#         y_w_cut[...,int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),:] = y_w_cut_inter
#         return y_w_cut
'''     
    def forward_chop_new(self, x, shave=12, batchsize = 64):
        h, w = x.size()[-2:]
        padsize = int(self.patch_size)
        shave = int(self.patch_size/4)

        scale = self.scale[self.idx_scale]

        h_cut = (h-padsize)%(padsize-shave)
        w_cut = (w-padsize)%(padsize-shave)

        x_unfold = torch.nn.functional.unfold(x, padsize, stride=padsize-shave).transpose(0,2).contiguous()

        x_hw_cut = x[...,(h-padsize):,(w-padsize):]
        y_hw_cut = self.model.forward(x_hw_cut)

        x_h_cut = x[...,(h-padsize):,:]
        x_w_cut = x[...,:,(w-padsize):]
        y_h_cut = self.cut_h_new(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_cut = self.cut_w_new(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        x_h_top = x[...,:padsize,:]
        x_w_top = x[...,:,:padsize]
        y_h_top = self.cut_h_new(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_top = self.cut_w_new(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        x_unfold = x_unfold.view(x_unfold.size(0),-1,padsize,padsize)
        y_unfold = []

        x_range = x_unfold.size(0)//batchsize + (x_unfold.size(0)%batchsize !=0)
        for i in range(x_range):
            y_unfold.append(self.model.forward(x_unfold[i*batchsize:(i+1)*batchsize,...]))
        y_unfold = torch.cat(y_unfold,dim=0)

        y = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut)*scale,(w-w_cut)*scale), padsize*scale, stride=padsize*scale-shave*scale)

        y[...,:padsize*scale,:] = y_h_top
        y[...,:,:padsize*scale] = y_w_top

        y_unfold = y_unfold[...,int(shave/2*scale):padsize*scale-int(shave/2*scale),int(shave/2*scale):padsize*scale-int(shave/2*scale)].contiguous()
        y_inter = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), padsize*scale-shave*scale, stride=padsize*scale-shave*scale)
        y[...,int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)] = y_inter

        y = torch.cat([y[...,:y.size(2)-int((padsize-h_cut)/2*scale),:],y_h_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2)
        y_w_cat = torch.cat([y_w_cut[...,:y_w_cut.size(2)-int((padsize-h_cut)/2*scale),:],y_hw_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2)
        y = torch.cat([y[...,:,:y.size(3)-int((padsize-w_cut)/2*scale)],y_w_cat[...,:,int((padsize-w_cut)/2*scale+0.5):]],dim=3)

        return y
    
    def cut_h_new(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        
        x_h_cut_unfold = torch.nn.functional.unfold(x_h_cut, padsize, stride=padsize-shave).transpose(0,2).contiguous()
        
        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0),-1,padsize,padsize)
        x_range = x_h_cut_unfold.size(0)//batchsize + (x_h_cut_unfold.size(0)%batchsize !=0)
        y_h_cut_unfold=[]
        for i in range(x_range):
            y_h_cut_unfold.append(self.model.forward(x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...]))
        y_h_cut_unfold = torch.cat(y_h_cut_unfold,dim=0)
        
        y_h_cut = torch.nn.functional.fold(y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),(padsize*scale,(w-w_cut)*scale), padsize*scale, stride=padsize*scale-shave*scale)
        y_h_cut_unfold = y_h_cut_unfold[...,:,int(shave/2*scale):padsize*scale-int(shave/2*scale)].contiguous()
        y_h_cut_inter = torch.nn.functional.fold(y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),(padsize*scale,(w-w_cut-shave)*scale), (padsize*scale,padsize*scale-shave*scale), stride=padsize*scale-shave*scale)
        y_h_cut[...,:,int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)] = y_h_cut_inter
        return y_h_cut
        
    def cut_w_new(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        
        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=padsize-shave).transpose(0,2).contiguous()
        
        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0),-1,padsize,padsize)
        x_range = x_w_cut_unfold.size(0)//batchsize + (x_w_cut_unfold.size(0)%batchsize !=0)
        y_w_cut_unfold=[]
        for i in range(x_range):
            y_w_cut_unfold.append(self.model.forward(x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...]))
        y_w_cut_unfold = torch.cat(y_w_cut_unfold,dim=0)
        
        y_w_cut = torch.nn.functional.fold(y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut)*scale,padsize*scale), padsize*scale, stride=padsize*scale-shave*scale)
        y_w_cut_unfold = y_w_cut_unfold[...,int(shave/2*scale):padsize*scale-int(shave/2*scale),:].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut-shave)*scale,padsize*scale), (padsize*scale-shave*scale,padsize*scale), stride=padsize*scale-shave*scale)
        y_w_cut[...,int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),:] = y_w_cut_inter
        return y_w_cut
'''
