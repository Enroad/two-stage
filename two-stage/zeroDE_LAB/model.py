import torch
import torch.nn as nn
import numpy as np
import BasicBlocks
import torch.nn.functional as F
def reflection(im):
    mr, mg, mb = torch.split(im, 1, dim=1)
    r = mr / (mr + mg + mb + 0.0001)
    g = mg / (mr + mg + mb + 0.0001)
    b = mb / (mr + mg + mb + 0.0001)
    return torch.cat([r, g, b], dim=1)


def luminance(s):
    return ((s[:, 0, :, :] + s[:, 1, :, :] + s[:, 2, :, :])).unsqueeze(1)

class light_net(nn.Module):
    def __init__(self):
        super(light_net, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        number_f = 32
        self.e_conv1 = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
        self.e_conv1_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv1_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv2_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv2_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv3_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv4_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv5_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv6_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv6_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv7_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv7_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        # self.ab_conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        # self.ab_relu1 = nn.ReLU()
        # self.ab_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.ab_relu2 = nn.ReLU()
        # self.ab_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.ab_relu3 = nn.ReLU()
        # self.ab_conv4 = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xo,attention_xo):
        # print(xo.type,attention_xo.type)
        # xo_addatt=xo+attention_xo
        xo_addatt = xo
        # xo_addatt = torch.cat([xo, attention_xo], 1)
        # xo_l_channel, xo_ab_channel = xo[:, 0:1, :, :], xo[:, 1:3, :, :]
        l_channel, ab_channel = xo_addatt[:, 0:1, :, :], xo_addatt[:, 1:3, :, :]
        x1 = self.relu(self.e_conv1(l_channel))
        x1_a = self.tanh(self.e_conv1_a(x1))
        x1_b = self.sigmoid(self.e_conv1_a(x1)) * 0.5 + 0.5

        x2 = self.relu(self.e_conv2(x1))
        x2_a = self.tanh(self.e_conv2_a(x2))
        x2_b = self.sigmoid(self.e_conv2_b(x2)) * 0.5 + 0.5

        x3 = self.relu(self.e_conv3(x2))
        x3_a = self.tanh(self.e_conv3_a(x3))
        x3_b = self.sigmoid(self.e_conv3_b(x3)) * 0.5 + 0.5

        x4 = self.relu(self.e_conv4(x3))
        x4_a = self.tanh(self.e_conv4_a(x4))
        x4_b = self.sigmoid(self.e_conv4_b(x4)) * 0.5 + 0.5

        x5 = self.relu(self.e_conv5(x4))
        x5_a = self.tanh(self.e_conv5_a(x5))
        x5_b = self.sigmoid(self.e_conv5_b(x5)) * 0.5 + 0.5

        x6 = self.relu(self.e_conv6(x5))
        x6_a = self.tanh(self.e_conv6_a(x6))
        x6_b = self.sigmoid(self.e_conv6_b(x6)) * 0.5 + 0.5

        x7 = self.relu(self.e_conv7(x6))
        x7_a = self.tanh(self.e_conv7_a(x7))
        x7_b = self.sigmoid(self.e_conv7_b(x7)) * 0.5 + 0.5
        xr = torch.cat([x1_a, x2_a, x3_a, x4_a, x5_a, x6_a, x7_a], dim=1)  # , x6_a, x7_a
        xr1 = torch.cat([x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b], dim=1)  # , x6_b, x7_b
        for i in np.arange(7):
            # xo = xo + xr[:, 3 * i:3 * i + 3, :, :] * torch.maximum(xo * (xr1[:, 3 * i:3 * i + 3, :, :] - xo) * (1 / xr1[:, 3 * i:3 * i + 3, :, :]),0*xo)
            l_channel = l_channel + xr[:, 3 * i:3 * i + 1, :, :] * 1 / (
                    1 + torch.exp(-10 * (-l_channel + xr1[:, 3 * i:3 * i + 1, :, :] - 0.1))) * l_channel * (
                                xr1[:, 3 * i:3 * i + 1, :, :] - l_channel) * (1 / xr1[:, 3 * i:3 * i + 1, :, :])

        # ab_channel = self.ab_relu1(self.ab_conv1(ab_channel))
        # ab_channel = self.ab_relu2(self.ab_conv2(ab_channel))
        # ab_channel = self.ab_relu3(self.ab_conv3(ab_channel))
        # ab_channel = self.ab_conv4(ab_channel)

        out = torch.cat((l_channel, ab_channel), dim=1)
        xr = torch.cat((xr, ab_channel), dim=1)
        xr1 = torch.cat((xr1, ab_channel), dim=1)


        # out =self.e_conv8(out)
        # print(out.shape,xr.shape,xr1.shape)
        return out,xr,xr1
