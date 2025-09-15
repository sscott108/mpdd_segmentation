import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNetAE(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # encoder
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*4, base*8)
        # decoder
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.uconv3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.uconv2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.uconv1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.p1(c1))
        c3 = self.d3(self.p2(c2))
        b = self.bottleneck(self.p3(c3))
        x = self.u3(b); x = self.uconv3(torch.cat([x, c3], dim=1))
        x = self.u2(x); x = self.uconv2(torch.cat([x, c2], dim=1))
        x = self.u1(x); x = self.uconv1(torch.cat([x, c1], dim=1))
        return self.out(x)