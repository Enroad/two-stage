import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np

# def rgb_to_grayscale(s):
#     return (0.2989*s[:,0,:,:]+ 0.5870*s[:,1,:,:] + 0.1140*s[:,2,:,:]).unsqueeze(1)
def rgb_to_grayscale(s):
    return ((s[:,0,:,:]+ s[:,1,:,:] +s[:,2,:,:])/3).unsqueeze(1)

def gradient2(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,:,4:,:] - img[:,:,:height-4,:]).abs()
    gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h*gradient2_h, gradient_w*gradient2_w

class L_SMO(nn.Module):
    def __init__(self, l_weight=1.0, a_weight=0, b_weight=0.0):
        super(L_SMO, self).__init__()
        self.l_weight = l_weight
        self.a_weight = a_weight
        self.b_weight = b_weight


    def forward(self, x_lab):
        batch_size = x_lab.size()[0]
        h_x = x_lab.size()[2]
        w_x = x_lab.size()[3]
        count_h = (x_lab.size()[2] - 1) * x_lab.size()[3]
        count_w = x_lab.size()[2] * (x_lab.size()[3] - 1)

        # Extract L, a, and b channels
        l_channel = x_lab[:, :1, :, :]
        a_channel = x_lab[:, 1:2, :, :]
        b_channel = x_lab[:, 2:, :, :]

        # Calculate TV loss for L channel
        h_tv_l = torch.pow((l_channel[:, :, 1:, :] - l_channel[:, :, :h_x - 1, :]), 2).sum()
        w_tv_l = torch.pow((l_channel[:, :, :, 1:] - l_channel[:, :, :, :w_x - 1]), 2).sum()

        # Calculate TV loss for a and b channels
        h_tv_a = torch.pow((a_channel[:, :, 1:, :] - a_channel[:, :, :h_x - 1, :]), 2).sum()
        w_tv_a = torch.pow((a_channel[:, :, :, 1:] - a_channel[:, :, :, :w_x - 1]), 2).sum()

        h_tv_b = torch.pow((b_channel[:, :, 1:, :] - b_channel[:, :, :h_x - 1, :]), 2).sum()
        w_tv_b = torch.pow((b_channel[:, :, :, 1:] - b_channel[:, :, :, :w_x - 1]), 2).sum()

        # Sum TV loss for all channels
        h_tv = self.l_weight * (h_tv_l / count_h) + self.a_weight * (h_tv_a / count_h) + self.b_weight * (
                    h_tv_b / count_h)
        w_tv = self.l_weight * (w_tv_l / count_w) + self.a_weight * (w_tv_a / count_w) + self.b_weight * (
                    w_tv_b / count_w)

        return 2 * (h_tv + w_tv) / batch_size

# SELF-DACE
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x_lab, org_lab):
        mean0 = self.pool(org_lab)
        ml0, ma0, mb0 = torch.split(mean0, 1, dim=1)
        xv0 = 1 - torch.abs(ml0)
        xv0 = xv0
        mean = self.pool(x_lab[:, 0:1, :, :])

        xv =mean
        dp = xv -  self.mean_val * xv0
        d = torch.pow(dp, 2)# * (1 + torch.pow(200, dp))
        return d
#ZERODCE
# class L_exp(nn.Module):
#
#     def __init__(self,patch_size,mean_val):
#         super(L_exp, self).__init__()
#         # print(1)
#         self.pool = nn.AvgPool2d(patch_size)
#         self.mean_val = mean_val
#     def forward(self, x_lab, org_lab ):
#         x=x_lab[:, 0:1, :, :]
#         x = torch.mean(x,1,keepdim=True)
#         mean = self.pool(x)
#         d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ),2))
#         return d


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape
        org_mean = torch.mean(org[:, :1, :, :],1,keepdim=True)
        enhance_mean = torch.mean(enhance[:, :1, :, :],1,keepdim=True)

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        # weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        # E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

class L_sobel(nn.Module):

    def __init__(self, lambda_edge,lambda_gradient):
        super(L_sobel, self).__init__()
        self.lambda_gradient = lambda_gradient
        self.lambda_edge=lambda_edge
    def forward(self, enhance_lab):
        output_l = enhance_lab[:, 0, :, :]


        output_dx = torch.abs(output_l[:, :, :-1] - output_l[:, :, 1:])
        output_dy = torch.abs(output_l[:, :-1, :] - output_l[:, 1:, :])


        gradient_loss = torch.mean(output_dx) + torch.mean(output_dy)


        edge_loss = torch.mean(torch.abs(output_l[:, :, :-1] - output_l[:, :, 1:])) + \
                    torch.mean(torch.abs(output_l[:, :-1, :] - output_l[:, 1:, :]))

        total_loss = self.lambda_edge * edge_loss + self.lambda_gradient * gradient_loss

        return total_loss


def Laplacian(x):
    #
    x = x.unsqueeze(0).unsqueeze(0)

    weight = torch.tensor([
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]],
        [[[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]]],
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]]
    ]).float().cuda()

    #
    weight = weight.unsqueeze(1).repeat(1, x.size(1), 1, 1, 1)

    frame = nn.functional.conv3d(x, weight, bias=None, stride=1, padding=1, dilation=1, groups=x.size(1))

    return frame.squeeze(0).squeeze(0)


def edge(x, imitation):
    def inference_mse_loss(frame_hr, frame_sr):
        content_base_loss = torch.mean(torch.sqrt((frame_hr - frame_sr) ** 2 + (1e-3) ** 2))
        return torch.mean(content_base_loss)

    x_edge = Laplacian(x)
    imitation_edge = Laplacian(imitation)
    edge_loss = inference_mse_loss(x_edge, imitation_edge)

    return edge_loss


class edgeloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image):
        loss = edge(out_image[:, 0, :, :], gt_image[:, 0, :, :])

        return loss
