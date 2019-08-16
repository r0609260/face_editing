

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
import numpy as np

from normalization import SPADE
# from face_editing.normalization import SPADE


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', n_downsampling=4):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                   norm_layer(ngf), nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                        norm_layer(ngf * mult * 2),
                        nn.ReLU(True)]

        self.encoder = nn.Sequential(*encoder)


        decoder = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            decoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                    use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                           padding=1, output_padding=1, bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]

        self.decoder = nn.Sequential(*decoder)

        output_layer = []

        output_layer += [nn.ReflectionPad2d(3)]
        output_layer += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        output_layer += [nn.Tanh()]

        self.output_layer = nn.Sequential(*output_layer)

    def forward(self, input, c):

        if self.input_nc > 3:
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, input.size(2), input.size(3))
            input = torch.cat([input, c], dim=1)

        x = self.encoder(input)
        # embeddings = x

        x = self.decoder(x)
        # feature_map = x

        rgb_image = self.output_layer(x)
        return rgb_image


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class View(nn.Module):
    def __init__(self, C, H, W):
        super(View, self).__init__()
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        if self.H is None and self.W is None:
            x = x.view(x.size(0), self.C)
        elif self.W is None:
            x = x.view(x.size(0), self.C, self.H)
        else:
            x = x.view(x.size(0), self.C, self.H, self.W)
        return x


class NLayerDiscriminator(nn.Module):

    def __init__(self, image_size=128, conv_dim=64, repeat_num=6):
        super(NLayerDiscriminator, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)

        # conv1 = list()
        # conv1.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        # conv1.append(nn.AdaptiveAvgPool2d([4, 4]))
        # self.conv1 = nn.Sequential(*conv1)

        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src


class MultiTaskDiscriminator(nn.Module):

    def __init__(self, image_size=128, conv_dim=64, repeat_num=6, c_dim=8):
        super(MultiTaskDiscriminator, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        conv1 = list()
        conv1.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        conv1.append(nn.AdaptiveAvgPool2d([4, 4]))
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


if __name__ == '__main__':

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    x = torch.rand((3, 3, 256, 256))
    model = MultiTaskDiscriminator(image_size=256, repeat_num=4)
    real, attr = model(x)
    print(real.size())
    print(attr.size())
