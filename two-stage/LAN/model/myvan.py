import torch.nn as nn
import torch
import model.BasicBlocks as BasicBlocks
import torch.nn.functional as F

#2.048097,64feature_num
#1.285082,32feature_num
#0.323505,16feature_num
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
class VisualAttentionNetwork(nn.Module):
    def __init__(self):
        super(VisualAttentionNetwork, self).__init__()

        self.feature_num = 64

        self.res_input_conv = nn.Sequential(
            DepthWiseConv(3, 16)  # 6
        )

        self.res_encoder1 = nn.Sequential(
            DepthWiseConv(16, 16),
            BasicBlocks.Residual_Block_New(16, 16, 3),
        )

        self.down1 = DownSample(16)

        self.res_encoder2 = nn.Sequential(
            DepthWiseConv(16, 32),
            BasicBlocks.Residual_Block_New(32, 32, 2),
        )

        self.down2 = DownSample(32)

        self.res_encoder3 = nn.Sequential(
            DepthWiseConv(32, 64),
            BasicBlocks.Residual_Block_New(64, 64, 1),
        )

        self.res_decoder3 = nn.Sequential(
            DepthWiseConv(64, 64),
            BasicBlocks.Residual_Block_New(64, 64, 1),
        )
        self.up2 = UpSample(64)

        self.res_decoder2 = nn.Sequential(
            DepthWiseConv(64, 32),
            BasicBlocks.Residual_Block_New(32, 32, 2),
        )
        self.up1 = UpSample(32)

        self.res_decoder1 = nn.Sequential(
            DepthWiseConv(32, 16),
            BasicBlocks.Residual_Block_New(16, 16, 3),
        )

        self.res_final = DepthWiseConv(16, 3)

        # self.AttentionNet = AttenteionNet()

    def forward(self, x, only_attention_output=False):
        res_input = self.res_input_conv(x)

        encoder1 = self.res_encoder1(res_input)
        encoder1_down = self.down1(encoder1)
        #
        encoder2 = self.res_encoder2(encoder1_down)
        encoder2_down = self.down2(encoder2)

        encoder3 = self.res_encoder3(encoder2_down)

        decoder3 = self.res_decoder3(encoder3) + encoder3
        decoder3 = self.up2(decoder3, output_size=encoder2.size())

        decoder2 = self.res_decoder2(decoder3) + encoder2
        decoder2 = self.up1(decoder2, output_size=encoder1.size())

        decoder1 = self.res_decoder1(decoder2) + encoder1

        output = self.res_final(decoder1)

        return output
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        # self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out
# --- Upsampling block in GridDehazeNet  --- #
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        # self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out

