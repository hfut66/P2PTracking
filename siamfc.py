from __future__ import absolute_import, division  #绝对导入，导入精确除法 3/4 = 0.75 而不是0

from collections import namedtuple #命名元组

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #包含 torch.nn 库中所有函数，同时包含大量 loss 和 activation function
import torch.nn.init as init #参数初始化
import torch.optim as optim #更新参数的优化算法（SGD、AdaGrad、RMSProp、 Adam等）
from got10k.trackers import Tracker #跟踪器
from torch.optim.lr_scheduler import ExponentialLR #学习率---指数衰减


class SiamFC(nn.Module): #继承，子类 SiamFC 继承于父类 nn.Module

    def __init__(self): #初始化，self指实例化对象的本身，不管调用下面定义的哪个函数，都需要执行这里的初始化
        super(SiamFC, self).__init__() # SiamFC继承父类的属性和方法，并用父类的同样的方式进行初始化
        self.feature = nn.Sequential(    #模型/序列容器,按顺序包装多个网络层
            # conv1
            nn.Conv2d(3, 96, 11, 2), #输入数据通道数， 输出数据通道数， 卷积核大小， 步长
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
                          #每个batch中图像的通道数(特征数)，eps为稳定系数（分母不能趋近或取0,给分母加上的值。默认为1e-5）。
            nn.ReLU(inplace=True), #inplace=True,用输出的数据覆盖输入的数据；节省空间，此时两者共用内存；
            nn.MaxPool2d(3, 2), #池化窗口大小，步长
            # conv2
            nn.Conv2d(96, 256, 5, 1, groups=2), # groups 卷积核个数
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3, 1, groups=2))
        self._initialize_weights() #权重初始化

    def forward(self, z, x):
        z = self.feature(z) # exemplar image z, for example, face
                            # In experiments, we will simply use the initial appearance of the object as the exemplar
        x = self.feature(x) # candidate/search image  x, for example, include the person and background


        # fast cross correlation 互相关
        n, c, h, w = x.size()       # 获取 x 的维度 (batchSize, channels, height, width)
        x = x.view(1, n * c, h, w)  # view函数相当于numpy的reshape
        out = F.conv2d(x, z, groups=n)  # F.conv2d(img1, window , padding=padd, dilation=dilation, groups=channel)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        out = 0.001 * out + 0.0   #相似度得分图？

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # 如果m与nn.Conv2d的类型相同，则返回True，否则返回False
                init.kaiming_normal_(m.weight.data, mode='fan_out', #权重初始化，init.kaiming_normal_ 是kaiming正太分布
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TrackerSiamFC(Tracker):#作用就是传入video sequence 和first frame 中的ground truth bbox,然后通过模型，得到后续帧的目标位置

    def __init__(self, net_path=None, **kargs): #继承于Tracker，需要重写
        super(TrackerSiamFC, self).__init__(name='SiamFC', is_deterministic=True)
        self.cfg = self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamFC()
        if net_path is not None:
            self.net.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))
                                # torch.load()加载 训练好 的模型
                                # load_state_dict()是net的一个方法，是将torch.load()加载出来的数据加载到net中
                                # 返回的是一个OrderedDict
        self.net = self.net.to(self.device) #模型加载到GPU或CPU上(数据加载给网络net，网络加载给设备gpu/cpu)

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        self.lr_scheduler = ExponentialLR(               # ExponentialLR指数衰减调整学习率
            self.optimizer, gamma=self.cfg.lr_decay)

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # inference parameters
            'exemplar_sz': 127, #模板尺寸，z
            'instance_sz': 255, #实例尺寸，x
            'context': 0.5, #填充参数
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,#缩放的惩罚因子
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            'adjust_scale': 0.001,
            # train parameters
            'initial_lr': 0.01,
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kargs.items():  #字典.items()返回的是可遍历的（键，值）元组数组
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)


    def _crop_and_resize(self, image, center, size, out_size, pad_color):
                        # 在imag图像上，以center为中心，crop(裁剪)出边长为size的正方形patch，然后再将resize成out_size大小
                        # 原始center = （y, x）
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),  #左上角corners
            np.round(center - (size - 1) / 2) + size)) #右下角corners
        corners = np.round(corners).astype(int)#corner = （y_min, x_min, y_max, x_max）

        # pad image if necessary
        pads = np.concatenate((-corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int) #pad后新的corner = （y_min, x_min, y_max, x_max）
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch


    def init(self, image, box):#传入第一帧的图片和标签(bbox)，初始化一些参数，计算一些之后搜索区域的中心等
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]，# box[l, t, w, h]==>box[centre_y, centre_x, h, w]
        box = np.array([ box[1]-1 + (box[3]-1) / 2, box[0]-1 + (box[2]-1) / 2, box[3], box[2]], dtype=np.float32)

        self.center, self.target_sz = box[:2], box[2:] #box的中心center=[centre_y, centre_x]，bbox宽高size=[h, w]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz  #响应图上采样后的大小 16*17
        self.hann_window = np.outer(np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()# 创建汉宁窗，也叫余弦窗，论文中说是增加惩罚。
            # 余弦窗就是为了解决边界效应，而解决的方法就是在目标原始像素上乘一个余弦窗使接近边缘的像素值接近于零。
            # Online, ... and a cosine window is added to the score map to penalize large displacements
            # np.outer（a, b） 1.对于多维向量，先转化成一维向量，2.第一个参数a表示倍数，使得第二个参数b每次变为几倍，
            # np.outer（a, b）= a转置乘以b

        # search scale factors    return  1.0375 **（-1.5,0,1.5）   3个值
        self.scale_factors = self.cfg.scale_step ** np.linspace( #等差数列
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num) #论文中提到两个变体，一个是5个尺度，一个是3个尺度（这里就是），1.0375 **（-1.5,0,1.5）

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz) # 1/2 *（h + w）
        # context就是边界语义信息，为了计算z_sz和x_sz，
        # 最后送入crop_and_resize去抠出搜索区域。最后抠出z_sz大小的作为exemplar image，并送入backbone，输出embedding，
        # 也可以看作是一个固定的互相关kernel，为了之后的相似度计算用，如论文中提到：
        # We found that updating (the feature representation of) the exemplar online through simple strategies, such as
        # linear interpolation, does not gain much performance and thus we keep it fixed
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))# self.target_sz矩阵每个元素都加context，再相乘在开方
        self.x_sz = self.z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image  对原始的图片数据进行处理，最后得到的才是 实例图像z
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            pad_color=self.avg_color)  #均值填充

        # exemplar features [H,W,C] ->[C,H,W]
        exemplar_image = torch.from_numpy(exemplar_image).to(self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel = self.net.feature(exemplar_image)

    def update(self, image):#就是传入当前帧，然后根据SiamFC网络更新目标的bbox坐标，之后就是根据这些坐标来show
        image = np.asarray(image)

        # search images
        instance_images = [self._crop_and_resize(   #在这新的帧里抠出search images，根据之前init里生成的3个尺度，
        # 然后resize成255×255，特别一点，我们可以发现search images在resize之前的边长x_sz大约为target_sz的4倍，
        # 这也印证了论文中的：we only search for the object within a region of approximately four times its previous size
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            pad_color=self.avg_color) for f in self.scale_factors]
        instance_images = np.stack(instance_images, axis=0)#[3, 255, 255, 3]，将这3个尺度的patch（也就是3个搜索范围）拼接一起送入self.net.feature
        instance_images = torch.from_numpy(instance_images).to(self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()
            instances = self.net.feature(instance_images)
            responses = F.conv2d(instances, self.kernel) * 0.001 #送入self.net.feature后，生成emdding后与之前的kernel进行互
            # 相关，得到score map，这些tensor的shape代码里都有标注，得到3个17×17的responses，然后对每一个response进行上采样到272×272
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty #此行和下面的一个=行就是对尺度进行惩罚，我是这样理解的，
        # 因为中间的尺度肯定是接近于1，其他两边的尺度不是缩一点就是放大一点，所以给以惩罚，如论文中说：Any change in scale is penalized
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))#之后就选出这3个通道里面最大的那个，并就行归一化和余弦窗惩罚，
                                                # 然后通过numpy.unravel_index找到一张response上峰值点(peak location)
        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - self.upscale_sz // 2 #接下来的问题就是：在response图中找到峰值点，那这在原图img中在
        # 哪里呢？所以我们要计算位移(displacement)，因为我们原本都是以目标为中心的，认为最大峰值点应该在response的中心，所以本220行就是峰
        # 值点和response中心的位移。
        disp_in_instance = disp_in_response * self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * self.scale_factors[scale_id] / self.cfg.instance_sz
        #因为之前在img上crop下一块instance patch，然后resize，然后
        # 送入CNN的backbone，然后score map又进行上采样成response，所以要根据这过程，逆回去计算对应在img上的位移，所以220-227行
        # 就是在做这件事
        self.center += disp_in_image #根据disp_in_image修正center，然后update target size，因为论文有一句：
        # update the scale by linear interpolation with a factor of 0.35 to provide damping

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([      #最后根据ops.show_image输入的需要，又得把bbox格式改回ltwh的格式
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box#return array

    def step(self, batch, backward=True, update_lr=False):
        if backward:
            self.net.train() #torch中 model.train() 训练
            if update_lr:
                self.lr_scheduler.step()
        else:
            self.net.eval() #torch中 model.eval() 测试

        z = batch[0].to(self.device)
        x = batch[1].to(self.device)

        with torch.set_grad_enabled(backward):#自动求导/梯度
            responses = self.net(z, x)
            labels, weights = self._create_labels(responses.size())
            loss = F.binary_cross_entropy_with_logits(
                responses, labels, weight=weights, size_average=True)

            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    def _create_labels(self, size): #设置标签
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels, self.weights

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos, np.ones_like(x), np.where(dist < r_neg, np.ones_like(x) * 0.5, np.zeros_like(x)))

            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        x, y = np.meshgrid(x, y) #坐标矩阵

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride # 16 / 8 = 2
        r_neg = self.cfg.r_neg / self.cfg.total_stride # 0 / 8 = 0
        labels = logistic_labels(x, y, r_pos, r_neg)

        # pos/neg weights
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights = np.zeros_like(labels)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        weights = weights.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))
        weights = np.tile(weights, [n, c, 1, 1]) #将weights在[n, c, 1, 1]四个维度上分别重复n,c,1,1次

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float() # torch.from_numpy(ndarray) → Tensor
        self.weights = torch.from_numpy(weights).to(self.device).float()

        return self.labels, self.weights