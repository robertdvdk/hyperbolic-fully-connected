from pathlib import Path
import sys
import torch

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from layers import Lorentz, Lorentz_Conv2d, Lorentz_fully_connected, LorentzBatchNorm


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, manifold: Lorentz):
        super(ResNetBlock, self).__init__()

        self.manifold = manifold
        stride = 1 if in_channels == out_channels else 2

        self.conv1 = Lorentz_Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, manifold=manifold, stride=stride, padding='same')
        self.bn1 = LorentzBatchNorm(manifold, num_features=out_channels)
        self.conv2 = Lorentz_Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, manifold=manifold, stride=1, padding='same')
        self.bn2 = LorentzBatchNorm(manifold, num_features=out_channels)

        if stride == 2:
            self.proj = Lorentz_Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, manifold=manifold, stride=stride, padding='same')
        else:
            self.proj = torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        x2 = self.conv1(x)
        # print("CONV1", x2[:, 0, :, :].mean())
        x2 = self.bn1(x2)
        # print("BN1", x2[:, 0, :, :].mean())
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        # print(x2[:, 0, :, :].mean())
        x = self.proj(x)
        # print(x[:, 0, :, :].mean())
        x = x.permute(0, 2, 3, 1)
        # x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x2 = x2.permute(0, 2, 3, 1).reshape(-1, x2.shape[1])
        x = self.manifold.expmap(x, 0.5 * self.manifold.logmap(x, x2))
        # x = x.reshape(x_shape)
        x = x.permute(0, 3, 1, 2)
        # print(x[:, 0, :, :].mean())
        return x


class ourModel(torch.nn.Module):
    def __init__(self, manifold: Lorentz):
        super(ourModel, self).__init__()

        self.manifold = manifold

        self.conv1 = Lorentz_Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding="same", bias=True, manifold=manifold)
        self.bn1 = LorentzBatchNorm(manifold, num_features=64)

        self.block2 = ResNetBlock(in_channels=64, out_channels=64, manifold=manifold)
        self.block3 = ResNetBlock(in_channels=64, out_channels=128, manifold=manifold)

        self.block4 = ResNetBlock(in_channels=128, out_channels=128, manifold=manifold)
        self.block5 = ResNetBlock(in_channels=128, out_channels=256, manifold=manifold)

        self.block6 = ResNetBlock(in_channels=256, out_channels=256, manifold=manifold)
        self.block7 = ResNetBlock(in_channels=256, out_channels=512, manifold=manifold)

        self.cls = Lorentz_fully_connected(in_features=512, out_features=100, manifold=manifold, do_mlr=True)

        # self.conv2 = Lorentz_Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold)
        # self.bn2 = LorentzBatchNorm(manifold, num_features=64)
        # self.conv3 = Lorentz_Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold)
        # self.bn3 = LorentzBatchNorm(manifold, num_features=64)

        # self.conv4 = Lorentz_Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding="same", bias=True, manifold=manifold)
        # self.bn4 = LorentzBatchNorm(manifold, num_features=128)
        # self.conv5 = Lorentz_Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold)
        # self.bn5 = LorentzBatchNorm(manifold, num_features=128)

        # self.conv6 = Lorentz_Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold)
        # self.bn2 = LorentzBatchNorm(manifold, num_features=64)
        # self.conv7 = Lorentz_Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold)
        # self.bn3 = LorentzBatchNorm(manifold, num_features=64)

        # self.conv3 = Lorentz_Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding="same", bias=True, manifold=manifold)
        # self.bn3 = LorentzBatchNorm(manifold, num_features=256)
        # self.conv4 = Lorentz_Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding="same", bias=True, manifold=manifold)
        # self.bn4 = LorentzBatchNorm(manifold, num_features=512)
        # # self.conv5 = Lorentz_Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding="same", bias=True, manifold=manifold)
        # # self.bn5 = LorentzBatchNorm(manifold, num_features=128)

        self.cls = Lorentz_fully_connected(in_features=512, out_features=100, manifold=manifold, do_mlr=True)
    
    
    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)

        
        x = self.block2(x)
        # print("============BLOCK2===============")
        # print(x.shape)
        
        x = self.block3(x)
        # print("============BLOCK3===============")
        # print(x.shape)
        
        x = self.block4(x)
        # print("============BLOCK4===============")
        # print(x.shape)
        x = self.block5(x)
        # print("============BLOCK5===============")
        # print(x.shape)
        
        x = self.block6(x)
        # print("============BLOCK6===============")
        # print(x.shape)
        x = self.block7(x)
        # print(x[:, 0, :, :].mean())
        # print("============BLOCK7===============")
        # x = self.manifold.expmap(x, 0.5 * self.manifold.logmap(x, x2))

        # x2 = self.conv2(x)
        # x2 = self.bn2(x2)
        # x = self.manifold.expmap(x, 0.5 * self.manifold.logmap(x, x2))

        # x2 = self.conv3(x)
        # x2 = self.bn3(x2)
        # x = self.manifold.expmap(x, 0.5 * self.manifold.logmap(x, x2))

        # x2 = self.conv4(x)
        # x2 = self.bn4(x2)
        # x = self.manifold.expmap(x, 0.5 * self.manifold.logmap(x, x2))

        # # x2 = self.conv5(x)
        # # x2 = self.bn5(x2)
        # x = self.manifold.expmap(x, 0.5 * self.manifold.logmap(x, x2))

        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # (B, H*W, C)
        x = self.manifold.lorentz_midpoint(x)
        # print("============FINAL===============")
        # print(x.shape)


        # x2 = self.fc1(x)
        # x = self.manifold.expmap(x, 0.5 * self.manifold.logmap(x, x2))

        # x = self.fc2(x2)
        # x = self.manifold.expmap(x, 0.5 * self.manifold.logmap(x, x2))

        x = self.cls(x)
        return x
