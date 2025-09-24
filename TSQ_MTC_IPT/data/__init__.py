import torch
import math
from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
from .bicubic import bicubic

'''这段代码主要用于构建和管理 PyTorch 中的数据加载器（DataLoader），
特别是针对分布式训练场景进行了优化，同时支持训练集和测试集的数据加载。'''

# This is a simple wrapper function for ConcatDataset
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args, world_size, rank):
        self.loader_train = None
        if not args.test_only:
            # datasets = []
            # for d in args.data_train:
            #     module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
            #     m = import_module('data.' + module_name.lower())
            #     datasets.append(getattr(m, module_name)(args, name=d))
            
            #import pdb; pdb.set_trace()
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                args.train_dataset,
                num_replicas=world_size,
                rank=rank)
            
            self.loader_train = dataloader.DataLoader(
                # MyConcatDataset(datasets),
                args.train_dataset,
                sampler=train_sampler,
                batch_size=args.batch_size,
                shuffle=False,
                # collate_fn=bicubic.forward,
                # pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
        
        # import pdb; pdb.set_trace()
        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109','CBSD68','Rain100L','GOPRO_Large']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)
            # test_sampler = SequentialDistributedSampler(testset, batch_size=args.test_batch_size, num_replicas=world_size, rank=rank)

            # self.loader_test.append(
            #     dataloader.DataLoader(
            #         testset,
            #         sampler=test_sampler,
            #         batch_size=args.test_batch_size,
            #         shuffle=False,
            #         # pin_memory=not args.cpu,
            #         num_workers=args.n_threads,
            #     )
            # )
            
            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=args.test_batch_size,
                    shuffle=False,
                )
            )
