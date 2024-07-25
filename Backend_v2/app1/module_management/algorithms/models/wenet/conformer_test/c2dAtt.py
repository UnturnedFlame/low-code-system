import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function




class C2DSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(C2DSELayer, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)#1 8->1 6
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, T, F = input_tensor.size()
        # Average along each time
        mean = input_tensor.permute(0, 2, 1, 3).mean(dim=1).unsqueeze(1) #(b, 1, c, f)
        #std
        squeeze_tensor = torch.sqrt(
            ((1/T) * (input_tensor.permute(0, 2, 1, 3) - mean).pow(2)).sum(dim=1).unsqueeze(1)
        )
        # excitation
        conv_out_1 = self.relu(self.bn(self.conv1(squeeze_tensor)))
        conv_out_2 = self.sigmoid(self.conv2(conv_out_1))#(b, 1, c, f)

        output_tensor = torch.mul(input_tensor, conv_out_2.permute(0, 2, 1, 3))
        return output_tensor


