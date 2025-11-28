import torch
import torch.nn as nn

# DiffusionModel class
class DiffusionModel():
    def __init__(self, n_step=1000, device='cuda'):
        self.n_step = n_step
        self.device = device

        betas = torch.linspace(1e-4, 0.02, n_step).to(device)
        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance_t = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    # q_sample function
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # p_loss function
    def p_loss(self, model, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = model(x_noisy, t)
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss

    # p_sample function
    def p_sample(self, model, x, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = (1. / self.alphas[t]).sqrt().view(-1, 1, 1, 1)

        mean = sqrt_recip_alphas_t * (x - (1 - alphas_t) / sqrt_one_minus_alphas_cumprod_t * model(x, t))

        if t[0] == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            posterior_variance_t = self.posterior_variance_t[t].view(-1, 1, 1, 1)

            return mean + posterior_variance_t.sqrt() * noise

    # sample function
    @torch.no_grad()
    def sample(self, model, batch_size, img_size, channels):
        img = torch.randn((batch_size, channels, img_size, img_size), device=self.device)
        imgs = []

        for i in reversed(range(0, self.n_step)):
            t_batch = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(model, img, t_batch)

            if i % (self.n_step // 7) == 0:
                img_converted = img.clamp(-1, 1)
                img_converted = (img_converted + 1) / 2
                imgs.append(img_converted.cpu())

        img = img.clamp(-1, 1)
        img = (img + 1) / 2

        return img, imgs

# Neural Network Components
class Sinusoidal_PosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, time):
        emb = time[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + self.residual_conv(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

# U-Net Model
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3
        out_channels = 3
        down_channels = [64, 128, 256]
        up_channels = [256, 128, 64]
        time_emb_dim = 256

        self.time_mlp = nn.Sequential(
            Sinusoidal_PosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        
        self.downs = nn.ModuleList()
        self.downs.append(nn.ModuleList([
            ResidualBlock(down_channels[0], down_channels[0], time_emb_dim),
            ResidualBlock(down_channels[0], down_channels[0], time_emb_dim),
            Downsample(down_channels[0], down_channels[1])
        ]))

        self.downs.append(nn.ModuleList([
            ResidualBlock(down_channels[1], down_channels[1], time_emb_dim),
            ResidualBlock(down_channels[1], down_channels[1], time_emb_dim),
            Downsample(down_channels[1], down_channels[2])
        ]))

        self.mid_block1 = ResidualBlock(256, 256, time_emb_dim)
        self.mid_block2 = ResidualBlock(256, 256, time_emb_dim)
        
        self.ups = nn.ModuleList()
        self.ups.append(nn.ModuleList([
            Upsample(up_channels[0], up_channels[1]),
            ResidualBlock(up_channels[0], up_channels[1], time_emb_dim),
            ResidualBlock(up_channels[1], up_channels[1], time_emb_dim)
        ]))

        self.ups.append(nn.ModuleList([
            Upsample(up_channels[1], up_channels[2]),
            ResidualBlock(up_channels[1], up_channels[2], time_emb_dim),
            ResidualBlock(up_channels[2], up_channels[2], time_emb_dim)
        ]))

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, up_channels[2]),
            nn.SiLU(),  
            nn.Conv2d(up_channels[2], out_channels, 1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.conv0(x)
        skip_connections = []

        for res1, res2, downsample in self.downs:
            x = res1(x, t)
            x = res2(x, t)
            skip_connections.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for upsample, res1, res2 in self.ups:
            x = upsample(x)
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = res1(x, t)
            x = res2(x, t)

        x = self.final_conv(x)

        return x