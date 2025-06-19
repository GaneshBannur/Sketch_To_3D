import torch
import torch.nn as nn
from torchvision.models import vgg16_bn

class SktEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16_bn(pretrained=True)
        self.encoder = nn.Sequential(*vgg.features[:27])
        self.custom = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(3, stride=3)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.encoder(x)
        x = self.custom(x)
        x = self.pool(x)
        return x.view(x.size(0), 256, -1)

class SktPointEncoder(nn.Module):
    def __init__(self, num_points=256):
        super().__init__()
        self.num_points = num_points
        
        self.conv1 = nn.Sequential(nn.Conv1d(2, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(2, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(2, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU())
        
        self.combine_conv = nn.Sequential(
            nn.Conv1d(256, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        self.mlp = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        f1 = self.conv1(x)
        f3 = self.conv2(x)
        f5 = self.conv3(x)
        
        f_cat = torch.cat([f1, f3, f5], dim=1)
        f_high = self.combine_conv(f_cat)
        
        global_feat = torch.max(f_high, dim=2, keepdim=True)[0].expand(-1, -1, self.num_points)
        fused = torch.cat([f1, global_feat], dim=1)
        out = self.mlp(fused)
        return out.permute(0, 2, 1)

class RCCA(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.channels = channels

    def forward(self, x):
        Q = self.q(x).permute(0, 2, 1)
        K = self.k(x)
        attn = self.softmax(torch.bmm(Q, K) / (self.channels ** 0.5))
        V = self.v(x).permute(0, 2, 1)
        context = torch.bmm(attn, V).permute(0, 2, 1)
        return x + context

class MHA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        seq = x.permute(2, 0, 1)
        for layer in self.layers:
            seq, _ = layer(seq, seq, seq)
        return seq.permute(1, 2, 0)

class CFAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rcca = RCCA(512)
        self.mha = MHA(512, 4, 6)

    def forward(self, Ft, Fp):
        fin = torch.cat([Ft, Fp], dim=1)
        out = self.rcca(fin)
        return self.mha(out)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512 * 64, 512 * 4 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(512, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x = self.fc(x)
        x = x.view(B, 512, 4, 4, 4)
        return self.deconv(x)

class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 32, 4, padding=2),
            nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.MaxPool3d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 4, padding=2),
            nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
            nn.MaxPool3d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, 4, padding=2),
            nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.MaxPool3d(2)
        )
        
        self.fc_sem = nn.Sequential(
            nn.Linear(128*4*4*4, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 8192),
            nn.LeakyReLU(0.2)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        B = x.size(0)
        sem = self.fc_sem(e3.view(B, -1)).view(B, 128, 4, 4, 4)
        x = e3 + sem
        
        d1 = self.up1(x)
        d1 = torch.cat([d1, e2], dim=1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        
        return self.up3(d2)

class MonoSketch3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = SktEncoder()
        self.point_enc = SktPointEncoder()
        self.fusion = CFAM()
        self.decoder = Decoder()
        self.refiner = Refiner()

    def forward(self, img, pts):
        img_feat = self.img_enc(img)
        pts_feat = self.point_enc(pts)
        fused = self.fusion(img_feat, pts_feat)
        coarse = self.decoder(fused)
        refined = self.refiner(coarse)
        return coarse, refined