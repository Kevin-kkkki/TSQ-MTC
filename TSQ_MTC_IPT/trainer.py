import utility  # 导入工具函数模块
import torch  # 导入PyTorch库
from tqdm import tqdm  # 导入进度条工具
import numpy as np  # 导入NumPy库
import copy  # 导入复制模块
from data.bicubic import bicubic  # 从数据模块导入双三次插值函数
import loss  # 导入损失函数模块
import os  # 导入操作系统模块
import dist_util  # 导入分布式工具模块
from model.lsq_plus import ActLSQ  # 从模型模块导入激活量化类

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args  # 保存命令行参数
        self.scale = args.scale  # 保存缩放比例

        self.ckp = ckp  # 保存检查点对象
        self.loader_train = loader.loader_train  # 保存训练数据加载器
        self.loader_test = loader.loader_test  # 保存测试数据加载器
        self.model = my_model  # 保存模型
        self.loss = my_loss  # 保存损失函数
        self.use_con = True  # 是否使用对比损失
        self.optimizer = utility.make_optimizer(args, self.model)  # 创建优化器
        self.last_epoch = None  # 最后一个epoch
        self.bicubic = bicubic(args)  # 初始化双三次插值对象
        self.error_last = 1e8  # 上次误差初始值
        self.criterion = my_loss  # 保存损失函数
        
        # 初始化对比损失
        self.con_loss = loss.SupConLoss()
        # self.ssim_loss = loss.SSIMLoss()  # SSIM损失（注释掉了）
    
    
    def pretrain(self, epoch):
        """预训练函数"""
        self.model.train()  # 设置模型为训练模式
        self.optimizer.zero_grad()  # 清空梯度
        tmp_epoch_losses = 0  # 临时保存epoch的总损失
        batch_idx = 0  # 批次索引
        
        print('数据长度: ', len(self.loader_train))  # 打印训练数据长度
        for batch_idx, imgs in enumerate(self.loader_train):  # 遍历训练数据
            # if batch_idx >= 0:
                # break  # 调试用，限制批次数量
            
            lr, hr, _ = imgs  # 解包低分辨率图像、高分辨率图像和其他信息
            idx_scale = self.loader_train.dataset.idx_scale  # 获取当前缩放比例索引
            lr, hr = self.prepare(lr, hr)  # 准备数据（转换设备和精度）
            
            if self.use_con:  # 如果使用对比损失
                ### 转换数据类型
                hr = hr.to(torch.float32)
                lr = lr.to(torch.float32)
                # 前向传播，获取超分辨率结果和中间特征
                sr, x_con = self.model(lr, idx_scale, self.use_con)
                loss1 = self.criterion(sr, hr)  # 计算主要损失（如MSE）
                loss2 = self.con_loss(x_con)  # 计算对比损失
                # loss3 = self.ssim_loss(sr, hr)  # SSIM损失（注释掉了）
                loss = loss1 + 0.1 * loss2  # 总损失 = 主要损失 + 0.1*对比损失
                tmp_epoch_losses += loss  # 累加损失
            
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播计算梯度
            
            self.optimizer.step()  # 更新参数
            
            # 定期打印训练信息
            if batch_idx % self.args.log_frequency == 0:
                print('Epoch: {}, Step: {}, Task: {}, 临时平均损失: {:.4f}, 总损失: {:.4f}, 损失1: {:.4f}, 损失2: {:.4f}'.format(
                    epoch, batch_idx, idx_scale, 
                    tmp_epoch_losses / (batch_idx + 1), 
                    loss, loss1, loss2
                ))
        
        self.optimizer.schedule()  # 调整学习率
        self.last_epoch = epoch  # 更新最后一个epoch

    def test(self, args):
        """测试函数"""
        torch.set_grad_enabled(False)  # 禁用梯度计算

        epoch = self.optimizer.get_last_epoch()  # 获取当前epoch
        self.ckp.write_log('\n评估:')  # 写入日志
        # 初始化日志记录
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()  # 设置模型为评估模式
        timer_test = utility.timer()  # 初始化计时器
        if self.args.save_results:  # 如果需要保存结果
            self.ckp.begin_background()  # 开始后台保存线程
        
        # 遍历测试数据集
        for idx_data, d in enumerate(self.loader_test):
            i = 0  # 计数器
            # 遍历缩放比例
            for idx_scale, scale in enumerate(self.scale):
                if idx_scale != args.set_task:  # 只测试指定的任务
                    continue
                d.dataset.set_scale(idx_scale)  # 设置数据集的缩放比例
                
                if self.args.derain:  # 如果是去雨任务
                    # 遍历测试数据
                    for norain, rain, filename in tqdm(d, ncols=80):
                        norain, rain = self.prepare(norain, rain)  # 准备数据
                        sr = self.model(rain, idx_scale)  # 模型预测
                        sr = utility.quantize(sr, self.args.rgb_range)  # 量化输出
                        
                        save_list = [sr, rain, norain]  # 要保存的图像列表
                        # 计算并累加PSNR
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, norain, scale, self.args.rgb_range
                        )
                        if self.args.save_results:  # 保存结果
                            self.ckp.save_results(d, filename[0], save_list, 1)
                    
                    # 计算平均PSNR
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)  # 找到最佳结果
                    # 写入日志
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (最佳: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                    isderain = 0  # 去雨标识
                
                elif self.args.denoise:  # 如果是去噪任务
                    # 遍历测试数据
                    for hr, _, filename in tqdm(d, ncols=80):
                        hr = self.prepare(hr)[0]  # 准备高分辨率图像
                        noisy_level = self.args.sigma  # 噪声水平
                        # 生成噪声并添加到高分辨率图像上
                        noise = torch.randn(hr.size()).mul_(noisy_level).cuda()
                        nois_hr = (noise + hr).clamp(0, 255)  # 带噪声的图像
                        sr = self.model(nois_hr, idx_scale)  # 模型预测去噪结果
                        sr = utility.quantize(sr, self.args.rgb_range)  # 量化输出

                        save_list = [sr, nois_hr, hr]  # 要保存的图像列表
                        # 计算并累加PSNR
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        if self.args.save_results:  # 保存结果
                            self.ckp.save_results(d, filename[0], save_list, 50)

                    # 计算平均PSNR
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)  # 找到最佳结果
                    # 写入日志
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (最佳: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                
                else:  # 默认是超分辨率任务
                    # 遍历测试数据
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)  # 准备数据
                        sr = self.model(lr, idx_scale)  # 模型预测超分辨率结果
                        sr = utility.quantize(sr, self.args.rgb_range)  # 量化输出

                        save_list = [sr]  # 要保存的图像列表
                        # 计算并累加PSNR
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        
                        if self.args.save_gt:  # 如果需要保存真值
                            save_list.extend([lr, hr])  # 添加低分辨率和高分辨率图像

                        if self.args.save_results:  # 保存结果
                            self.ckp.save_results(d, filename[0], save_list, scale)
                        i += 1  # 计数器加1
                    
                    # 计算平均PSNR
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)  # 找到最佳结果
                    
                    if self.last_epoch is None:  # 如果是只测试模式
                        self.last_epoch = 0
                        print('现在是只测试模式 !!!!!!')
                    
                    # 写入日志
                    self.ckp.write_log(
                        '[{} x{}]\tepoch:{}\t PSNR : {:.3f} (最佳: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.last_epoch + 1,
                            self.ckp.log[-1, idx_data, idx_scale],  # 当前epoch的PSNR
                            best[0][idx_data, idx_scale],  # 最佳PSNR
                            best[1][idx_data, idx_scale] + 1  # 最佳epoch索引
                        )
                    )
        
        # 记录测试时间
        self.ckp.write_log('前向传播时间: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('保存中...')

        if self.args.save_results:  # 如果保存了结果
            self.ckp.end_background()  # 结束后台保存线程

        # 记录总时间
        self.ckp.write_log(
            '总时间: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)  # 重新启用梯度计算

    def prepare(self, *args):
        """准备数据：转换设备和精度"""
        device = torch.device('cpu' if self.args.cpu else 'cuda')  # 选择设备
        
        def _prepare(tensor):
            """内部函数：处理单个张量"""
            if self.args.precision == 'half':  # 如果需要半精度
                tensor = tensor.half()
            return tensor.to(device)  # 移动到指定设备

        return [_prepare(a) for a in args]  # 处理所有输入的张量

    def terminate(self):
        """判断是否终止训练"""
        if self.args.test_only:  # 如果是只测试模式
            self.test()  # 执行测试
            return True
        else:  # 训练模式
            epoch = self.optimizer.get_last_epoch() + 1  # 获取当前epoch
            return epoch >= self.args.epochs  # 判断是否达到最大epoch数
    
    def _np2Tensor(self, img, rgb_range):
        """将NumPy数组转换为Tensor"""
        np_transpose = np.ascontiguousarray(img.transpose((2, 1, 2)))  # 转换通道顺序
        tensor = np_transpose.astype(np.float32)  # 转换为float32
        tensor = tensor * (rgb_range / 255)  # 归一化到指定的RGB范围
        return tensor





# import utility
# import torch
# from tqdm import tqdm
# import numpy as np
# import copy
# from data.bicubic import bicubic
# import loss
# import os
# import dist_util
# from model.lsq_plus import ActLSQ

# class Trainer():
#     def __init__(self, args, loader, my_model, my_loss, ckp):
#         self.args = args
#         self.scale = args.scale

#         self.ckp = ckp
#         self.loader_train = loader.loader_train
#         self.loader_test = loader.loader_test
#         self.model = my_model
#         self.loss = my_loss
#         self.use_con = True
#         self.optimizer = utility.make_optimizer(args, self.model)
#         self.last_epoch = None
#         self.bicubic = bicubic(args)
#         self.error_last = 1e8
#         self.criterion = my_loss
        
#         # import pdb; pdb.set_trace()
#         self.con_loss = loss.SupConLoss()
#         # self.ssim_loss = loss.SSIMLoss()
    
    
#     def pretrain(self, epoch):
    
#         self.model.train()
#         self.optimizer.zero_grad()
#         # import pdb; pdb.set_trace()
#         tmp_epoch_losses = 0
#         batch_idx = 0
        
#         # import pdb; pdb.set_trace()
#         print('data length: ', len(self.loader_train))
#         for batch_idx, imgs in enumerate(self.loader_train):
            
#             # if batch_idx >= 0:
#                 # break
            
#             # import pdb; pdb.set_trace()
#             lr, hr, _ = imgs
#             idx_scale = self.loader_train.dataset.idx_scale
#             lr, hr = self.prepare(lr, hr)
#             if self.use_con:
                
#                 ### to dtype
#                 hr = hr.to(torch.float32)
#                 lr = lr.to(torch.float32)
#                 sr, x_con = self.model(lr, idx_scale, self.use_con)
#                 loss1 = self.criterion(sr, hr)
#                 loss2 = self.con_loss(x_con)
#                 # loss3 = self.ssim_loss(sr, hr)
#                 loss = loss1 + 0.1 * loss2
#                 tmp_epoch_losses += loss
            
#             self.optimizer.zero_grad()
#             loss.backward()
            
#             self.optimizer.step()
            
#             if batch_idx % self.args.log_frequency == 0:
#                 print('Epoch: {}, Step: {}, Task: {}, tmp_epoch_avg_loss: {:.4f}, loss: {:.4f}, loss1: {:.4f}, loss2: {:.4f}'.format(epoch, batch_idx, idx_scale, tmp_epoch_losses / (batch_idx + 1), loss, loss1, loss2))
            
#         self.optimizer.schedule()
#         self.last_epoch = epoch

#     def test(self, args):
#         # import pdb; pdb.set_trace()
#         torch.set_grad_enabled(False)

#         epoch = self.optimizer.get_last_epoch()
#         self.ckp.write_log('\nEvaluation:')
#         self.ckp.add_log(
#             torch.zeros(1, len(self.loader_test), len(self.scale))
#         )
#         self.model.eval()
#         timer_test = utility.timer()
#         if self.args.save_results: self.ckp.begin_background()
#         for idx_data, d in enumerate(self.loader_test):
#             # import pdb; pdb.set_trace()
#             i = 0
#             for idx_scale, scale in enumerate(self.scale):
#                 if idx_scale != args.set_task:
#                     continue
#                 d.dataset.set_scale(idx_scale)
#                 if self.args.derain:
#                     for norain, rain, filename in tqdm(d, ncols=80):
#                         norain, rain = self.prepare(norain, rain)
#                         sr = self.model(rain, idx_scale)
#                         sr = utility.quantize(sr, self.args.rgb_range)
                        
#                         save_list = [sr, rain, norain]
#                         self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
#                             sr, norain, scale, self.args.rgb_range
#                         )
#                         if self.args.save_results:
#                             self.ckp.save_results(d, filename[0], save_list, 1)
#                     self.ckp.log[-1, idx_data, idx_scale] /= len(d)
#                     best = self.ckp.log.max(0)
#                     self.ckp.write_log(
#                         '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
#                             d.dataset.name,
#                             scale,
#                             self.ckp.log[-1, idx_data, idx_scale],
#                             best[0][idx_data, idx_scale],
#                             best[1][idx_data, idx_scale] + 1
#                         )
#                     )
#                     isderain = 0
#                 elif self.args.denoise:
#                     for hr, _,filename in tqdm(d, ncols=80):
#                         hr = self.prepare(hr)[0]
#                         noisy_level = self.args.sigma
#                         noise = torch.randn(hr.size()).mul_(noisy_level).cuda()
#                         nois_hr = (noise+hr).clamp(0,255)
#                         sr = self.model(nois_hr, idx_scale)
#                         sr = utility.quantize(sr, self.args.rgb_range)

#                         save_list = [sr, nois_hr, hr]
#                         self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
#                             sr, hr, scale, self.args.rgb_range
#                         )
#                         if self.args.save_results:
#                             self.ckp.save_results(d, filename[0], save_list, 50)

#                     self.ckp.log[-1, idx_data, idx_scale] /= len(d)
#                     best = self.ckp.log.max(0)
#                     self.ckp.write_log(
#                         '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
#                             d.dataset.name,
#                             scale,
#                             self.ckp.log[-1, idx_data, idx_scale],
#                             best[0][idx_data, idx_scale],
#                             best[1][idx_data, idx_scale] + 1
#                         )
#                     )
#                 else:
#                     for lr, hr, filename in tqdm(d, ncols=80):
#                         lr, hr = self.prepare(lr, hr)
#                         sr = self.model(lr, idx_scale)
#                         sr = utility.quantize(sr, self.args.rgb_range)

#                         save_list = [sr]
#                         self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
#                             sr, hr, scale, self.args.rgb_range
#                         )
                        
#                         if self.args.save_gt:
#                             save_list.extend([lr, hr])

#                         if self.args.save_results:
#                             self.ckp.save_results(d, filename[0], save_list, scale)
#                         i = i+1
#                     # import pdb; pdb.set_trace()
#                     self.ckp.log[-1, idx_data, idx_scale] /= len(d)
#                     best = self.ckp.log.max(0)
                    
#                     if self.last_epoch is None:
#                         self.last_epoch = 0
#                         print('Now is the test_only mode !!!!!!')
                    
#                     self.ckp.write_log(
#                         '[{} x{}]\tepoch:{}\t PSNR : {:.3f} (Best: {:.3f} @epoch {})'.format(
#                             d.dataset.name,
#                             scale,
#                             self.last_epoch + 1,
#                             self.ckp.log[-1, idx_data, idx_scale], ### tmp epoch acc
#                             best[0][idx_data, idx_scale], ### best epoch acc
#                             best[1][idx_data, idx_scale] + 1 ### epoch idx
#                         )
#                     )
                    
#                     ### save checkpoint every args.save_every epoch
#                     # if epoch % self.args.save_every == 0:
#                     #     print("Saving checkpoint of epoch {}...".format(self.last_epoch + 1))
#                     #     tmp_save_path = self.args.save + 'epoch_{}.pt'.format(self.last_epoch + 1)
#                     #     torch.save(self.model.state_dict(), tmp_save_path)
#                     ### save checkpoint if tmp epoch obtains the best acc.
#                     # if self.ckp.log[-1, idx_data, idx_scale] >= best[0][idx_data, idx_scale]:
#                     #     print("Saving best epoch at epoch {}...".format(self.last_epoch + 1))
#                     #     tmp_save_path = self.args.save + 'best_epoch.pt'
#                     #     torch.save(self.model.state_dict(), tmp_save_path)
                    
#         # import pdb; pdb.set_trace()
#         self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
#         self.ckp.write_log('Saving...')

#         if self.args.save_results:
#             self.ckp.end_background()

#         self.ckp.write_log(
#             'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
#         )

#         torch.set_grad_enabled(True)

#     def prepare(self, *args):
#         device = torch.device('cpu' if self.args.cpu else 'cuda')
#         def _prepare(tensor):
#             if self.args.precision == 'half': tensor = tensor.half()
#             return tensor.to(device)

#         return [_prepare(a) for a in args]

#     def terminate(self):
#         if self.args.test_only:
#             # import pdb; pdb.set_trace()
#             self.test()
#             return True
#         else:
#             epoch = self.optimizer.get_last_epoch() + 1
#             return epoch >= self.args.epochs
#     def _np2Tensor(self, img, rgb_range):
#         np_transpose = np.ascontiguousarray(img.transpose((2, 1, 2)))
#         tensor = np_transpose.astype(np.float32)
#         tensor = tensor * (rgb_range / 255)
#         return tensor
