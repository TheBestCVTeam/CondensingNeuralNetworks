import math

import torch
import torch.fft
import torch.nn as nn
import numpy as np

BN = nn.BatchNorm2d


# noinspection PySingleQuotedDocstring
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# noinspection PyAbstractClass
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# noinspection PyAbstractClass
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# AENet_C,S,G is based on ResNet-18
# noinspection PyAbstractClass,PyListCreation,PyDefaultArgument,
# PyUnusedLocal,PyUnusedLocal
# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,
# PyUnusedLocal
# noinspection PyAbstractClass,PyUnusedLocal,PyUnusedLocal
class AENet(nn.Module):

    # noinspection PyDefaultArgument,PyDefaultArgument,PyUnusedLocal,
    # PyUnusedLocal
    # noinspection PyDefaultArgument,PyDefaultArgument,PyUnusedLocal
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2],
                 blocks_sizes=[64, 128, 256, 512], num_cnn_layers=4,
                 num_classes=1000, sync_stats=False, relevant_layers=True,
                 dl_bypass_input: bool = False):

        global BN
        self.relevant_layers = relevant_layers
        self.inplanes = 64
        self.num_cnn_layers = num_cnn_layers
        self.dl_bypass_input = dl_bypass_input

        # model parameters
        super(AENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # to vary the number of cnn layers
        if self.num_cnn_layers < 1 or type(self.num_cnn_layers) != int:
            raise ValueError("num_cnn_layers must be an integer >= 1")

        if self.num_cnn_layers >= 1:
            self.layer1 = self._make_layer(block, 64, layers[0])

        if self.num_cnn_layers >= 2:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if self.num_cnn_layers >= 3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        if self.num_cnn_layers == 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.max_block_size = blocks_sizes[num_cnn_layers - 1]

        # The average pooling layer needs a kernel size to vary based on the
        # output size of the last convolutional layer
        # a 1 layer cnn has an output of (channel, 64, 56, 56), the last 2
        # dimensions decrease by a factor of 2^n
        # where n is the depth of the cnn layer
        # (i.e. a cnn layer with a depth of 4 has the dimentions (channels,
        # 512, 7, 7)
        self.avgpool_k_shape = int(56 / 2 ** (num_cnn_layers - 1))
        self.avgpool = nn.AvgPool2d(self.avgpool_k_shape, stride=1)

        if not self.relevant_layers:
            # Three classifiers of semantic informantion
            self.fc_live_attribute = nn.Linear(
                self.max_block_size * block.expansion, 40)
            self.fc_attack = nn.Linear(self.max_block_size * block.expansion,
                                       11)
            self.fc_light = nn.Linear(self.max_block_size * block.expansion, 5)

        # One classifier of Live/Spoof information
        fc_live_input_size = self.max_block_size * block.expansion
        if self.dl_bypass_input:
            #TODO: make this a variable
            fc_live_input_size += 224 ** 2 # flattened (224, 224) vector

        self.fc_live = nn.Linear(fc_live_input_size, 2)

        if not self.relevant_layers:
            # Two embedding modules of geometric information
            self.upsample14 = nn.Upsample((14, 14), mode='bilinear')
            self.depth_final = nn.Conv2d(self.max_block_size, 1, kernel_size=3,
                                         stride=1, padding=1, bias=False)
            self.reflect_final = nn.Conv2d(self.max_block_size, 3,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)
            # The ground truth of depth map and reflection map has been
            # normalized[torchvision.transforms.ToTensor()]
            self.sigmoid = nn.Sigmoid()

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # noinspection PyListCreation
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def set_train_cnn_layers(self, train=True):
        # if True, cnn layers will be trained during the training step
        # if false, the weights and bias of cnn layers will not be changed
        if self.num_cnn_layers >= 1:
            self.layer1.training = train

        if self.num_cnn_layers >= 2:
            self.layer2.training = train

        if self.num_cnn_layers >= 3:
            self.layer3.training = train

        if self.num_cnn_layers == 4:
            self.layer4.training = train

    def forward(self, x):

        # generate fourier for image
        if self.dl_bypass_input:
            f_vec = generate_fourier(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.num_cnn_layers >= 1:
            x = self.layer1(x)

        if self.num_cnn_layers >= 2:
            x = self.layer2(x)

        if self.num_cnn_layers >= 3:
            x = self.layer3(x)

        if self.num_cnn_layers == 4:
            x = self.layer4(x)

        if not self.relevant_layers:
            depth_map = self.depth_final(x)
            reflect_map = self.reflect_final(x)

            depth_map = self.sigmoid(depth_map)
            # noinspection PyUnusedLocal
            depth_map = self.upsample14(depth_map)

            reflect_map = self.sigmoid(reflect_map)
            reflect_map = self.upsample14(reflect_map)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if not self.relevant_layers:
            x_live_attribute = self.fc_live_attribute(x)
            x_attack = self.fc_attack(x)
            x_light = self.fc_light(x)

        # rejoin vector with cnn output tensor
        # keep in mind that this tensor has been flattened after the last cnn layer
        # so the input vecor will be appended to the end

        if self.dl_bypass_input:
            # vec_channel = torch.zeros()
            # x = x_in[:,:,0:2] # extract image from input tensor, the image will be the first 3 channels
            # vec_channel = x_in[:,:,3] # fourth channel is reserved for vector input
            # vec = vec_channel[~ torch.isnan(vec_channel)] # extract only the values that are not np.NaN
            x = torch.cat((x, f_vec), 1)

        try:
            x_live = self.fc_live(x)
        except Exception as e:
            print(f"error training dense layer: {e}")
            return x
        return x_live

def generate_fourier(x):
    x = torch.fft.fftn(x[:,0,:,:], s=(224, 224)).abs()  # move to device, e.g. GPU
    x = x.view(x.shape[0], -1)  # "flatten"
    #x = torch.unsqueeze(x, 2)
    return x

if __name__ == "__main__":
    model = AENet(num_classes=2, num_cnn_layers=4, relevant_layers=True, dl_bypass_input = True)
    model_ft = model.to(torch.device('cpu'))
    # model_ft.set_train_cnn_layers(False)
    # checkpoint = torch.load('../pickle/ckpt_iter.pth.tar',
    #                         map_location={'cuda:0': 'cpu'})
    rand_im = torch.rand(1,3,250,250) # let image be a 300 x 325 x 3 random tensor
    vec = torch.rand(20, 1)# let vec be a 20 x 1 random vector


    print(model)

    # generate vector channel
    vec_ch = torch.zeros(rand_im.shape[2], rand_im.shape[3])
    vec_ch[:] = np.nan
    vec_ch[0:vec.size()[0], 0] = vec[:, 0]


    # add both image and vector channel to input
    full_input = torch.zeros(rand_im.shape[0], rand_im.shape[1] + 1, rand_im.shape[2], rand_im.shape[3])
    full_input[:,0:3,:,:] = rand_im[0:3,:,:]
    full_input[:,3,:,:] = vec_ch
    model.forward(full_input)
    print("done")