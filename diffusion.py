import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalNoisePredictor(nn.Module):
    def __init__(self, x_dim, cond_dim, edge_type_dim, t_dim=32):
        super().__init__()
        self.t_dim = t_dim
        
        # Nhúng thời gian (Time embedding)
        self.t_mlp = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.GELU(),
            nn.Linear(t_dim, t_dim)
        )
        
        input_dim = x_dim + cond_dim + edge_type_dim + t_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, x_dim)
        )
        
    def forward(self, x, t, c, edge_type):
        t_emb = self.t_mlp(t.float().unsqueeze(-1))
        h = torch.cat([x, c, edge_type, t_emb], dim=-1)
        return self.net(h)

class ConditionalDDPM(nn.Module):
    def __init__(self, predictor, num_steps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.predictor = predictor
        self.num_steps = num_steps
        
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
    def forward(self, x_0, c, edge_type):
        t = torch.randint(0, self.num_steps, (x_0.size(0),), device=x_0.device)
        noise = torch.randn_like(x_0)
        
        x_t = self.q_sample(x_0, t, noise=noise)
        noise_pred = self.predictor(x_t, t, c, edge_type)
        
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, c, edge_type, shape):
        device = c.device
        b_size = c.size(0)
        
        x_t = torch.randn(*shape, device=device)
        
        for i in reversed(range(self.num_steps)):
            t = torch.full((b_size,), i, device=device, dtype=torch.long)
            noise_pred = self.predictor(x_t, t, c, edge_type)
            
            beta_t = self.betas[t].unsqueeze(-1)
            alpha_t = self.alphas[t].unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            
            mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred)
            
            if i > 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(beta_t)
                x_t = mean + sigma * noise
            else:
                x_t = mean
                
        # Sigmoid để đưa feature về lại domain [0, 1] như lúc tạo bằng ICVAE
        return torch.sigmoid(x_t)


class Diffuser(nn.Module):
    def __init__(self, x_dim, cond_dim, edge_type_dim, num_steps=100):
        super().__init__()
        predictor = ConditionalNoisePredictor(x_dim, cond_dim, edge_type_dim)
        self.ddpm = ConditionalDDPM(predictor, num_steps=num_steps)
        self.latent_size = 50 # Để tương thích với API của ICVAE (nếu model gọi logic tương tự)
        self.x_dim = x_dim
        
    def forward(self, x, c, edge_type):
        loss = self.ddpm(x, c, edge_type)
        return loss
        
    def inference(self, dummy_z, c, edge_type):
        """
        Sử dụng dummy_z chỉ để có chung format gọi hàm với ICVAE
        """
        shape = (c.shape[0], self.x_dim)
        return self.ddpm.sample(c, edge_type, shape)
