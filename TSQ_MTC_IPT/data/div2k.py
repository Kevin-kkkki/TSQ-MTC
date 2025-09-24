import os
from data import srdata

# 定义DIV2K类，继承自srdata.SRData，用于处理DIV2K数据集
class DIV2K(srdata.SRData):
    # 初始化方法，接收参数对象args，数据集名称name，是否为训练集train，是否为基准测试benchmark
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        # import pdb; pdb.set_trace()
        # 解析数据范围参数：将args.data_range按'/'分割后，再将每个部分按'-'分割
        # 例如"1-800/801-810"会被解析为[[1,800], [810,810]]
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]                                      # 训练集使用第一部分范围
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]                                  # 测试集且仅有一部分范围时使用该范围
            else:
                data_range = data_range[1]                                  # 如果有两部分，则用第二部分

        self.begin, self.end = list(map(lambda x: int(x), data_range))      # 将数据范围转换为整数，得到开始和结束索引
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        #import pdb; pdb.set_trace()
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        #import pdb; pdb.set_trace()
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        #if self.input_large: self.dir_lr += 'L'

