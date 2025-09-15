import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)
    )

class SharedUNetAE(nn.Module):
    """
    One-class U-Net autoencoder shared across metals.
    FiLM at bottleneck conditions on metal_id: b' = gamma * b + beta
    """
    def __init__(self, in_ch=3, base=64, out_ch=3, num_metals=0, emb_dim=32):
        super().__init__()
        # Encoder
        self.pool = nn.MaxPool2d(2,2)
        self.down1 = double_conv(in_ch, base)
        self.down2 = double_conv(base, base*2)
        self.down3 = double_conv(base*2, base*4)
        self.down4 = double_conv(base*4, base*8)
        self.bottleneck = double_conv(base*8, base*8)

        # FiLM conditioning
        self.num_metals = num_metals
        C = base*8
        if num_metals > 0:
            self.metal_emb = nn.Embedding(num_metals, emb_dim)
            self.film = nn.Linear(emb_dim, 2*C)  # -> gamma,beta

        # Decoder
        # Decoder (keep your upsamples)
        self.up4  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv4 = double_conv(base*12, base*4)   #  up4(out)=base*4  +  skip x4=base*8  -> base*12

        self.up3  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv3 = double_conv(base*6,  base*2)   #  up3(out)=base*2  +  skip x3=base*4  -> base*6

        self.up2  = nn.ConvTranspose2d(base*2, base,   2, stride=2)
        self.conv2 = double_conv(base*3,  base)     #  up2(out)=base    +  skip x2=base*2  -> base*3

        self.up1  = nn.ConvTranspose2d(base,   base,   2, stride=2)
        self.conv1 = double_conv(base*2,  base)     #  up1(out)=base    +  skip x1=base    -> base*2

        self.out = nn.Conv2d(base, out_ch, 1)


    def encode(self, x):
        x1 = self.down1(x)              # B, B, H, W
        x2 = self.down2(self.pool(x1))  # B, 2B, H/2, W/2
        x3 = self.down3(self.pool(x2))  # B, 4B, H/4, W/4
        x4 = self.down4(self.pool(x3))  # B, 8B, H/8, W/8
        b  = self.bottleneck(self.pool(x4))  # B, 8B, H/16, W/16
        return x1, x2, x3, x4, b

    def apply_film(self, b, metal_ids):
        if self.num_metals == 0:
            return b
        # metal_ids: [B] (LongTensor)
        gamma_beta = self.film(self.metal_emb(metal_ids))  # [B, 2C]
        B, C, H, W = b.shape
        gamma, beta = gamma_beta[:, :C], gamma_beta[:, C:]  # [B,C]
        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)
        return gamma * b + beta

    def forward(self, x, metal_ids=None):
        x1, x2, x3, x4, b = self.encode(x)
        if metal_ids is None:
            # default to zeros to avoid crashing if num_metals>0 but id not given
            metal_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        b = self.apply_film(b, metal_ids)

        u = self.up4(b); u = self.conv4(torch.cat([u, x4], dim=1))
        u = self.up3(u); u = self.conv3(torch.cat([u, x3], dim=1))
        u = self.up2(u); u = self.conv2(torch.cat([u, x2], dim=1))
        u = self.up1(u); u = self.conv1(torch.cat([u, x1], dim=1))
        logits = self.out(u)
        recon  = torch.sigmoid(logits)  # 0..1 if inputs normalized to 0..1
        return recon
