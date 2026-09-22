from torch import nn
import torch

from utils.channels import CHANNELS, REGIONS, BANDS


class RegionCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(1, 16, (1, 25), padding=(0, 12)),
                                     nn.BatchNorm2d(16), nn.ELU(), nn.MaxPool2d((1, 5)))
        # Effective dilated kernel = 99; padding 49 preserves 500 samples.
        self.branch2 = nn.Sequential(nn.Conv2d(1, 16, (1, 50), dilation=(1, 2), padding=(0, 49)),
                                     nn.BatchNorm2d(16), nn.ELU(), nn.MaxPool2d((1, 5)))
        self.fusion = nn.Sequential(nn.Conv2d(32, 32, (channels, 1)), nn.BatchNorm2d(32),
                                   nn.ELU(), nn.MaxPool2d((1, 5)))

    def forward(self, x):
        return self.fusion(torch.cat((self.branch1(x), self.branch2(x)), dim=1)).flatten(1)


class SpectralFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 128))
        self.register_buffer("frequencies", torch.fft.rfftfreq(500, 1 / 250))

    def forward(self, x):
        # Linear-scale band energy; no Welch, log transform, or relative power.
        power = torch.fft.rfft(x, dim=-1).abs().square() / x.shape[-1]
        energy = torch.stack([power[..., (self.frequencies >= low) & (self.frequencies <= high)].sum(-1).mean(-1)
                              for low, high in BANDS], dim=-1)
        return self.encoder(energy)


class PatchEmbedding(nn.Module):
    def __init__(self, embedding=64):
        super().__init__()
        self.indices = {region: [CHANNELS.index(ch) for ch in channels] for region, channels in REGIONS.items()}
        self.region_cnns = nn.ModuleDict({region: RegionCNN(len(indices)) for region, indices in self.indices.items()})
        self.spectral = SpectralFeatureExtractor()
        self.projection = nn.Sequential(nn.Linear(32 * 20 + 128, embedding), nn.LayerNorm(embedding))
        self.position = nn.Parameter(torch.randn(5, embedding))

    def forward(self, x):
        tokens = []
        for region, indices in self.indices.items():
            regional = x[:, indices]
            value = torch.cat((self.region_cnns[region](regional.unsqueeze(1)), self.spectral(regional)), dim=1)
            tokens.append(self.projection(value))
        return torch.stack(tokens, dim=1) + self.position


class AttentionBlock(nn.Module):
    def __init__(self, dim=64, heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop1, self.drop2 = nn.Dropout(0.1), nn.Dropout(0.1)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * dim, dim))

    def forward(self, x, return_attention=False):
        update, weights = self.attention(x, x, x, need_weights=return_attention, average_attn_weights=False)
        x = self.norm1(x + self.drop1(update))
        return self.norm2(x + self.drop2(self.ff(x))), weights


class BRAINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding(64)
        self.blocks = nn.ModuleList([AttentionBlock(64, 8) for _ in range(4)])
        self.apen_embedding = nn.Sequential(nn.Linear(63, 252), nn.ReLU(), nn.Dropout(0.5))
        self.classifier = nn.Sequential(nn.Linear(5 * 64 + 252, 256), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(256, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 3))

    def forward(self, x, apen, return_details=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (63, 500):
            raise ValueError("Expected EEG shape (batch, 63, 500) in CHANNELS order")
        if apen is None or tuple(apen.shape) != (len(x), 63):
            raise ValueError("Expected channel-wise ApEn of shape (batch, 63)")
        value = self.patch_embedding(x)
        attention = []
        for block in self.blocks:
            value, weights = block(value, return_details)
            if weights is not None:
                attention.append(weights)
        embedding = torch.cat((value.flatten(1), self.apen_embedding(apen.float())), dim=1)
        logits = self.classifier(embedding)
        return (logits, embedding, attention) if return_details else logits
