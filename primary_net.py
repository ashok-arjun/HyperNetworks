import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hypernetwork_modules import HyperNetwork
from resnet_blocks import ResNetBlock


class Embedding(nn.Module):

    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim

        h,k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))
        
    def forward(self, hyper_net):
        ww = []
        h, k = self.z_num
        print("Embedding")
        print(h,k)
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j])) # Always should be of dim 64
            ww.append(torch.cat(w, dim=1))
        print(torch.cat(ww, dim=0).shape)
        return torch.cat(ww, dim=0)


class PrimaryNetwork(nn.Module):

    def __init__(self, z_dim=64):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.z_dim = z_dim 
        self.hope = HyperNetwork(z_dim=self.z_dim)

        # Pairs of zs are used as multipliers with default values, to calculate w1,w2 of a resnet block    

        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]] # the dimensions of the weight matrices to be generator

        self.res_net = nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        self.zs = nn.ModuleList()

        for i in range(36):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim))

        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64,10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        print(x.shape)

        for i in range(18):
            print(i)
            w1 = self.zs[2*i](self.hope)
            w2 = self.zs[2*i+1](self.hope)
            
            print(w1.shape, w2.shape)

            x = self.res_net[i](x, w1, w2)

            print(x.shape)

        print("finally")
        x = self.global_avg(x)

        print(x.shape)

        x = self.final(x.view(-1,64))

        print(x.shape)

        return x


# Generating two resnet weights at every instance
# Thye are generated inside the Embedding class
# It 


if __name__ == "__main__":
    net = PrimaryNetwork().cuda()
    x = torch.randn(8,3,32,32).cuda()
    output = net(x)
    print(output.shape)