from option import args

import torch
import utility
import data
import loss
from trainer import Trainer
import warnings
warnings.filterwarnings('ignore')
import os
import model
torch.manual_seed(args.seed)  # 设置随机种子
checkpoint = utility.checkpoint(args)  # 初始化检查点对象
from data.div2k import DIV2K

# 单卡调试开关
SINGLE_CARD_DEBUG = True  # True: VS Code 单卡调试, False: 多卡分布式训练

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import dist_util


def init_seed(seed=23) -> None:
    import random
    import numpy as np
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    global model
    init_seed(args.seed)

    if not SINGLE_CARD_DEBUG:
        # -----------------------
        # 多卡分布式训练初始化
        # -----------------------
        dist_util.init_distributed_mode(args)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        num_tasks = dist_util.get_world_size()
        global_rank = dist_util.get_rank()
    else:
        # 单卡调试模式
        num_tasks = 1
        global_rank = 0
        device = torch.device('cuda:0')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if checkpoint.ok:
        args.train_dataset = DIV2K(args)
        loader = data.Data(args, num_tasks, global_rank)

        state_dict = torch.load(args.pretrain, map_location='cpu')

        # -----------------------
        # 模型初始化 + DDP/单卡
        # -----------------------
        if SINGLE_CARD_DEBUG:
            # 单卡调试模式
            _model = model.Model(args, checkpoint).to(device)
            _model.load_state_dict(state_dict, strict=False)
        else:
            # 多卡分布式模式
            gpu_id = getattr(args, 'local_rank', 0)
            device = torch.device(f'cuda:{gpu_id}')
            _model = model.Model(args, checkpoint).to(device)
            _model.load_state_dict(state_dict, strict=False)
            _model = DDP(
                _model,
                device_ids=[gpu_id],
                output_device=gpu_id,
                find_unused_parameters=True
            )

        # 初始化损失函数（训练时）
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        # 初始化训练器
        t = Trainer(args, loader, _model, _loss, checkpoint)

        os.makedirs(args.save, exist_ok=True)

        # -----------------------
        # 执行训练或测试
        # -----------------------
        if not args.test_only:
            for epoch in range(args.epochs):
                t.pretrain(epoch)
                if SINGLE_CARD_DEBUG or dist_util.get_rank() == 0:
                    t.test(args)
            checkpoint.done()
        else:
            t.test(args)


if __name__ == '__main__':
    main()
