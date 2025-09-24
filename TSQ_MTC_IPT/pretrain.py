from option import args

import torch
import utility
import data
import loss
from trainer import Trainer
import warnings
warnings.filterwarnings('ignore')
import os
# os.system('pip install einops')
import model
torch.manual_seed(args.seed)                                        # 设置随机种子以确保结果可复现
checkpoint = utility.checkpoint(args)                               # 初始化检查点对象以管理模型保存和日志记录
from data.div2k import DIV2K

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import dist_util


def init_seed(seed=23) -> None:
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    
    import random
    import numpy as np
    os.environ['PYTHONHASHSEED'] = str(seed)                            # 设置Python哈希种子，影响字典等的随机性    
    random.seed(seed)                                                   # 设置random模块的随机种子
    np.random.seed(seed)                                                # 设置NumPy的随机种子                                                    
    torch.manual_seed(seed)                                             # 设置PyTorch的随机种子                                    
    torch.cuda.manual_seed(seed)                                        # 设置当前GPU的随机种子                      
    torch.cuda.manual_seed_all(seed)                                    # 设置所有GPU的随机种子                          


def main():
    global model                                                        # 声明全局变量model，方便在函数内访问和修改
    ## set ddp
    dist_util.init_distributed_mode(args)                               # 调用自定义工具初始化分布式模式，设置进程和GPU分配
    torch.backends.cudnn.enabled = True                                 # 启用CuDNN加速，提升卷积计算效率
    torch.backends.cudnn.benchmark = True                               # 启用CuDNN的自动调优功能，加速训练（输入尺寸固定时效果好）
    num_tasks = dist_util.get_world_size()                              # 获取分布式训练的总进程数
    global_rank = dist_util.get_rank()                                  # 获取当前进程的全局排名    
    
    init_seed()

    ### set here to enable test_only
    # args.test_only = True
    if checkpoint.ok:                                                   # 检查检查点工具是否初始化成功                         
        # import pdb; pdb.set_trace() 
        args.train_dataset = DIV2K(args)                                # 初始化DIV2K训练数据集实例             
        loader = data.Data(args, num_tasks, global_rank)                # 初始化数据加载器，传入参数和分布式信息
        
        state_dict = torch.load(args.pretrain, map_location='cpu')      # 加载预训练模型权重，先加载到CPU以避免GPU内存问题
        
        _model = model.Model(args, checkpoint)                          # 初始化模型实例，传入参数和检查点对象  
        # 将模型封装为DDP分布式模型，指定当前进程使用的GPU，允许存在未使用的参数          
        _model = DDP(_model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
        _model.load_state_dict(state_dict,strict=False)                 # 加载预训练权重

        # 初始化损失函数（仅训练时需要，测试时为None）
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        # 初始化训练器，传入配置、数据加载器、模型、损失函数和检查点工具
        t = Trainer(args, loader, _model, _loss, checkpoint)            

        #import pdb; pdb.set_trace()
        os.makedirs(args.save, exist_ok=True)                           # 创建保存结果的目录，若已存在则不报错

        if not args.test_only:                                          # 如果不是仅测试模式（即执行训练）                      
            for epoch in range(0, args.epochs):
                t.pretrain(epoch)                                       # 执行当前轮次的预训练
                if dist_util.get_rank() == 0:                           # 仅主进程（rank=0）执行测试，避免重复计算
                    t.test(args)
            checkpoint.done()                                           # 训练完成后标记检查点状态                          
        elif args.test_only:                                            # 如果是仅测试模式                     
            t.test(args)                                                # 执行测试                          
            
if __name__ == '__main__':
    main()
