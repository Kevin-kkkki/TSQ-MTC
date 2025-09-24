import argparse

# 创建解析器，描述为IPT
parser = argparse.ArgumentParser(description='IPT')

# --debug: 布尔参数，添加该参数时启用调试模式
parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')

# --template: 模板路径，默认值为当前目录，用于指定配置模板
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
# --n_threads: 数据加载线程数，默认8个线程
parser.add_argument('--n_threads', type=int, default=8,
                    help='number of threads for data loading')

# --cpu: 布尔参数，添加该参数时仅使用CPU运行（不使用GPU）
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')


# parser.add_argument('--n_GPUs', type=int, default=4, #1
#                     help='number of GPUs')
# --n_GPUs: 使用的GPU数量，默认2个
parser.add_argument('--n_GPUs', type=int, default=2, #1
                    help='number of GPUs')

# --seed: 随机种子，默认值为1，用于结果复现
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')






# Data specifications
# --dir_data: 数据集存放目录，默认'/cache/data/'
parser.add_argument('--dir_data', type=str, default='/cache/data/',
                    help='dataset directory')

# --dir_demo: 演示图像目录，默认'../test'
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')

# --data_train: 训练数据集名称，默认'Set5'
parser.add_argument('--data_train', type=str, default='Set5',
                    help='train dataset name') ### TO CHECK

# --data_test: 测试数据集名称，默认'DIV2K'
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name') ### TO CHECK # DIV2K

# --data_range: 训练/测试数据的范围，格式为"训练范围/测试范围"，默认'1-3550/3541-3550'
parser.add_argument('--data_range', type=str, default='1-3550/3541-3550',
                    help='train/test data range') ### TO CHECK ### 1-3550/3541-3550

# --ext: 数据集文件扩展名，默认'sep'（可能表示分离的文件格式）
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')

# --scale: 超分辨率缩放比例，默认2倍（可通过"+"分隔设置多个比例，如"2+4"）
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')

# --patch_size: 输出图像块的大小，默认48x48
parser.add_argument('--patch_size', type=int, default=48,
                    help='output patch size')

# --rgb_range: RGB值的最大值，默认255（表示0-255的像素范围）
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')

# --n_colors: 颜色通道数，默认3（表示RGB图像）
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

# --no_augment: 布尔参数，添加该参数时不使用数据增强
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# --set_task: 评估任务的缩放比例，默认0（可能表示不指定特定任务）
parser.add_argument('--set_task', type=int, default='0',
                    help='scale set to eval')

# Model specifications
# --model: 模型名称，默认'qipt'（指定使用的模型架构）
parser.add_argument('--model', default='qipt',
                    help='model name') ### TO CHECK

# --n_feats: 特征图数量，默认64
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')

# --shift_mean: 是否从输入中减去像素均值，默认True
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')

# --precision: 测试时的浮点精度，可选'single'（单精度）或'half'（半精度），默认'single'
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')





# Training specifications
# --reset: 布尔参数，添加该参数时重置训练（不加载之前的检查点）
parser.add_argument('--reset', action='store_true',
                    help='reset the training')

# --alltask: 布尔参数，添加该参数时预训练所有6个任务
parser.add_argument('--alltask', action='store_true',
                    help='pretrain all 6 tasks')

# --test_every: 每训练N个批次后进行一次测试，默认1000
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches') ### TO CHECK

# --epochs: 训练的轮数，默认50
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train') ### TO CHECK

# --batch_size: 训练时的批次大小，默认1
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training') ### TO CHECK

# --test_batch_size: 测试时的批次大小，默认1
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='input batch size for testing')

# --crop_batch_size: 训练时的裁剪批次大小，默认64（可能用于图像裁剪的数据加载）
parser.add_argument('--crop_batch_size', type=int, default=64,
                    help='input batch size for training')

# --split_batch: 将批次拆分为更小的块数，默认1（不拆分）
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')

# --self_ensemble: 布尔参数，添加该参数时在测试中使用自集成方法（提升性能）
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')

# --test_only: 是否仅进行测试（不训练），默认False
parser.add_argument('--test_only', type=bool, default=False,
                    help='set this option to test the model')
# parser.add_argument('--test_only', action='store_true',
#                     help='set this option to test the model')
# --gan_k: GAN损失中的k值（可能表示判别器更新次数），默认1
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')






# Optimization specifications
# --lr: 学习率，默认1e-5
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate') ### TO CHECK

# --decay: 学习率衰减类型，默认'200'（可能表示每200个epoch衰减一次）
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')

# --gamma: 学习率衰减因子，默认0.5（每次衰减时乘以该因子）
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')

# --optimizer: 优化器类型，可选SGD、ADAM、ADAMW、RMSprop，默认ADAMW
parser.add_argument('--optimizer', default='ADAMW',
                    choices=('SGD', 'ADAM','ADAMW', 'RMSprop'), ### TO CHECK ### ADAMW
                    help='optimizer to use (SGD | ADAM | RMSprop)')

# --momentum: SGD优化器的动量值，默认0.9
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')

# --betas: ADAM优化器的beta参数，默认(0.9, 0.999)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')

# --epsilon: ADAM优化器的数值稳定性参数，默认1e-8
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')

# --weight_decay: 权重衰减（L2正则化）系数，默认1e-4
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay') ### 1e-4

# --gclip: 梯度裁剪阈值，默认0（表示不进行裁剪）
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')






# Loss specifications
# --loss: 损失函数配置，默认'1*L1'（表示使用L1损失，权重为1）
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration') ### TO CHECK ### 1*Charbonnie

# --skip_threshold: 跳过批次的误差阈值，默认1e8（超过该误差的批次将被跳过）
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')





# Log specifications
# --save: 保存结果的文件名/目录，默认'results'
parser.add_argument('--save', type=str, default='results',
                    help='file name to save') ### TO CHECK

# --load: 加载模型的文件名，默认空（不加载）
parser.add_argument('--load', type=str, default='',
                    help='file name to load')

# --resume: 从特定检查点恢复训练，默认0（不恢复）
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')

# --save_models: 布尔参数，添加该参数时保存所有中间模型
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')

# --print_every: 每训练多少批次打印一次训练状态，默认100
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')

# --save_results: 布尔参数，添加该参数时保存输出结果
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

# --save_gt: 布尔参数，添加该参数时保存低分辨率和高分辨率图像
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

# --save_every: 每多少轮保存一次检查点，默认5
parser.add_argument('--save_every', type=int, default=5,
                    help='how many epochs to save checkpoint once')

# --log_frequency: 每多少步记录一次损失信息，默认100
parser.add_argument('--log_frequency', type=int, default=100, help='how many steps to log loss information once')

# --save_checkpoint_frequency: 每多少步保存一次检查点，默认500
parser.add_argument('--save_checkpoint_frequency', type=int, default=500, help='how many steps to save checkpoint once')


              

#cloud
# --moxfile: 可能与云服务（如阿里云MOX）相关的配置，默认1
parser.add_argument('--moxfile', type=int, default=1)

# --data_url: 数据集在云存储中的路径
parser.add_argument('--data_url', type=str,help='path to dataset')

# --train_url: 训练结果在云存储中的保存路径
parser.add_argument('--train_url', type=str, help='train_dir')

# --pretrain: 预训练模型的路径，默认空
parser.add_argument('--pretrain', type=str, default='') ### TO CHECK

# --load_query: 加载查询的配置，默认0
parser.add_argument('--load_query', type=int, default=0)




#transformer
# --patch_dim: 图像块的维度，默认3
parser.add_argument('--patch_dim', type=int, default=3)

# --num_heads: Transformer注意力头数，默认12
parser.add_argument('--num_heads', type=int, default=12)

# --num_layers: Transformer层数，默认12
parser.add_argument('--num_layers', type=int, default=12)

# --dropout_rate: Dropout概率，默认0（不使用dropout）
parser.add_argument('--dropout_rate', type=float, default=0)

# --no_norm: 布尔参数，添加该参数时不使用归一化层
parser.add_argument('--no_norm', action='store_true')

# --freeze_norm: 布尔参数，添加该参数时冻结归一化层参数
parser.add_argument('--freeze_norm', action='store_true')

# --post_norm: 布尔参数，添加该参数时使用Post-LayerNorm（而非Pre-LayerNorm）
parser.add_argument('--post_norm', action='store_true')

# --no_mlp: 布尔参数，添加该参数时不使用MLP层
parser.add_argument('--no_mlp', action='store_true')

# --pos_every: 布尔参数，添加该参数时在每个Transformer层都使用位置编码
parser.add_argument('--pos_every', action='store_true')

# --no_pos: 布尔参数，添加该参数时不使用位置编码
parser.add_argument('--no_pos', action='store_true')

# --num_queries: 查询向量的数量，默认6
parser.add_argument('--num_queries', type=int, default=6)





#denoise
# --denoise: 布尔参数，添加该参数时启用去噪功能
parser.add_argument('--denoise', action='store_true')
# --sigma: 噪声水平（用于去噪任务），默认30
parser.add_argument('--sigma', type=float, default=30)




#derain
# --derain: 布尔参数，添加该参数时启用去雨功能
parser.add_argument('--derain', action='store_true')

# --derain_test: 去雨测试的配置，默认1
parser.add_argument('--derain_test', type=int, default=1)

# --derain_dir: 去雨数据集的目录，默认空
parser.add_argument('--derain_dir', type=str, default='')





#deblur
# --deblur: 布尔参数，添加该参数时启用去模糊功能
parser.add_argument('--deblur', action='store_true')

# --deblur_test: 去模糊测试的配置，默认1
parser.add_argument('--deblur_test', type=int, default=1)

# quantization




# ddp
# --world_size: 分布式进程数量，默认1（不使用分布式训练）
parser.add_argument('--world_size',default=1, type=int, help='number of distributed processes')

# --dist_url: 用于设置分布式训练的URL，默认'env://'（从环境变量读取）
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

# 解析命令行参数，返回参数对象和未解析的参数
args, unparsed = parser.parse_known_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))                 # 将scale从字符串转为整数列表（支持多个缩放比例）  
args.data_train = args.data_train.split('+')                                    # 将训练数据集名称拆分为列表（支持多个数据集）
args.data_test = args.data_test.split('+')                                      # 将测试数据集名称拆分为列表（支持多个数据集）

    
if args.epochs == 0:
    args.epochs = 1e8                                                           # 如果epochs为0，则设置为一个很大的数，表示无限训练直到手动停止

# 将字符串形式的布尔值转换为实际的布尔类型
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
