"""bicubic"""
import numpy as np
import os
import imageio
import random
from PIL import Image

def search(root, target="JPEG"):
    """imagent"""
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            item_list.extend(search(path, target))
        elif path.split('.')[-1] == target:
            item_list.append(path)
        elif path.split('/')[-1].startswith(target):
            item_list.append(path)
    return item_list

def get_patch_img(img, patch_size=96, scale=2): # patch_size: 48(default); scale: 2 or 3 or 4 or 1, depending on the task idx
    """imagent"""
    ih, iw = img.shape[:2] ### 350, 500
    tp = scale * patch_size
    if (iw - tp) > -1 and (ih-tp) > 1: ### if patch size smaller than the image size, randomly crop a patch from the given image and return
        ix = random.randrange(0, iw-tp+1)
        iy = random.randrange(0, ih-tp+1)
        hr = img[iy:iy+tp, ix:ix+tp, :3]
    elif (iw - tp) > -1 and (ih - tp) <= -1: ### if patch size larger than the height, then randomly crop a patch with the whole height, and resize the patch to patch_size
        ix = random.randrange(0, iw-tp+1)
        hr = img[:, ix:ix+tp, :3]
        pil_img = Image.fromarray(hr).resize((tp, tp), Image.BILINEAR)
        hr = np.array(pil_img)
    elif (iw - tp) <= -1 and (ih - tp) > -1: ### if patch size larger than the wdith, then randomly crop a patch with the whole wdith, and resize the patch to patch_size
        iy = random.randrange(0, ih-tp+1)
        hr = img[iy:iy+tp, :, :3]
        pil_img = Image.fromarray(hr).resize((tp, tp), Image.BILINEAR)
        hr = np.array(pil_img)
    else: ### if both width and height smaller than the patch_size, then directly resize the image to the patch_size (may cause distortion)
        pil_img = Image.fromarray(img).resize((tp, tp), Image.BILINEAR)
        hr = np.array(pil_img)
    return hr

def random_sample_integers(start, end, m):
    # 生成整数范围列表
    integers = list(range(start, end+1))
    # 使用 random.sample() 函数随机抽取 m 次
    samples = random.sample(integers, m)
    return samples

class bicubic:
    """bicubic"""
    def __init__(self, args, seed=0):
        self.seed = seed
        self.args = args
        self.rand_fn = np.random.RandomState(self.seed)
        if len(args.scale) == 6:
            self.dataroot = self.args.dir_data
            self.derain_dataroot = self.args.derain_dir
            # self.derain_dataroot = os.path.join(self.dataroot, "RainTrainL")
            self.derain_img_list = search(self.derain_dataroot, "rainstreak")

    def cubic(self, x):
        absx2 = np.abs(x) * np.abs(x)
        absx3 = np.abs(x) * np.abs(x) * np.abs(x)

        condition1 = (np.abs(x) <= 1).astype(np.float32)
        condition2 = ((np.abs(x) > 1) & (np.abs(x) <= 2)).astype(np.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * np.abs(x) + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        """bicubic"""
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = np.arange(start=1, stop=out_size[0]+1).astype(np.float32)
        x1 = np.arange(start=1, stop=out_size[1]+1).astype(np.float32)

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = np.floor(u0 - kernel_width / 2)
        left1 = np.floor(u1 - kernel_width / 2)

        width = np.ceil(kernel_width) + 2
        indice0 = np.expand_dims(left0, axis=1) + \
                  np.expand_dims(np.arange(start=0, stop=width).astype(np.float32), axis=0)
        indice1 = np.expand_dims(left1, axis=1) + \
                  np.expand_dims(np.arange(start=0, stop=width).astype(np.float32), axis=0)

        mid0 = np.expand_dims(u0, axis=1) - np.expand_dims(indice0, axis=0)
        mid1 = np.expand_dims(u1, axis=1) - np.expand_dims(indice1, axis=0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (np.expand_dims(np.sum(weight0, axis=2), 2))
        weight1 = weight1 / (np.expand_dims(np.sum(weight1, axis=2), 2))

        indice0 = np.expand_dims(np.minimum(np.maximum(1, indice0), in_size[0]), axis=0)
        indice1 = np.expand_dims(np.minimum(np.maximum(1, indice1), in_size[1]), axis=0)
        kill0 = np.equal(weight0, 0)[0][0]
        kill1 = np.equal(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, imgs):
        """bicubic""" 
        hr, lrx2, lrx3, lrx4 = imgs ### 48*48, 96*96, 144*144, 192*192
        idx = self.rand_fn.randint(0, 6)
        if idx < 3:
            if idx == 0:
                scale = 1/2
                hr = lrx2 # if scale = 2/3/4, use the selected 96/144/192 as the high-resolution patch
            elif idx == 1:
                scale = 1/3
                hr = lrx3
            elif idx == 2:
                scale = 1/4
                hr = lrx4
            # import pdb; pdb.set_trace()
            hr = np.array(hr)
            [_, _, h, w] = hr.shape
            weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
            weight0 = np.asarray(weight0[0], dtype=np.float32)

            indice0 = np.asarray(indice0[0], dtype=np.float32).astype(np.long)
            weight0 = np.expand_dims(np.expand_dims(np.expand_dims(weight0, axis=0), axis=1), axis=4)
            out = hr[:, :, (indice0-1), :] * weight0
            out = np.sum(out, axis=3)
            A = np.transpose(out, (0, 1, 3, 2))

            weight1 = np.asarray(weight1[0], dtype=np.float32)
            weight1 = np.expand_dims(np.expand_dims(np.expand_dims(weight1, axis=0), axis=1), axis=4)
            indice1 = np.asarray(indice1[0], dtype=np.float32).astype(np.long)
            out = A[:, :, (indice1-1), :] * weight1
            out = np.round(255 * np.transpose(np.sum(out, axis=3), (0, 1, 3, 2)))/255
            out = np.clip(np.round(out), 0, 255)
            lr = out
            # hr = hr
        else:
            if idx == 4:
                hr = np.array(hr)
                noise = np.random.randn(*hr.shape) * 30
                lr = np.clip(noise + hr, 0, 255)
                # hr = list(hr)
                # lr = list(lr)
            elif idx == 5:
                hr = np.array(hr)
                noise = np.random.randn(*hr.shape) * 50
                lr = np.clip(noise + hr, 0, 255)
                # hr = list(hr)
                # lr = list(lr)
            elif idx == 3:
                hr = np.array(hr)
                rain = self._load_rain(hr.shape[0])
                lr = np.clip(hr+rain, 0, 255)
        # print(lr.shape)
        # print(hr.shape)
        return lr, hr, idx

    def _load_rain(self, bs):
        # idx = random.randint(0, len(self.derain_img_list) - 1)
        lrs = []
        idxs = random_sample_integers(0, len(self.derain_img_list) - 1, bs)
        for idx in idxs:
            f_lr = self.derain_img_list[idx]
            f_lr = imageio.imread(f_lr)
            rain = np.expand_dims(f_lr, axis=2)
            rain = self.get_patch(rain, 1)
            lrs.append(rain)
        
        stacked_images = np.stack(lrs, axis=0)

        rains = np.transpose(stacked_images, (0, 3, 1, 2))
        return rains
    
    def get_patch(self, lr, scale=0):
        if scale == 0:
            scale = self.scale[self.idx_scale]
        lr = get_patch_img(lr, patch_size=self.args.patch_size, scale=scale)
        return lr ### in fact, here return a high-resolution patch from tmp image

    def _np2Tensor(self, img, rgb_range):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = np_transpose.astype(np.float32)
        tensor = tensor * (rgb_range / 255)
        if tensor.shape[0] == 4:
            tensor = tensor[:3]
        return tensor
