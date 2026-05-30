
import torch
from torch import nn

from network.blocks_and_layers.blocks import EncoderLayer, DecoderLayer
from network.embedding.embedding import TransformerEmbedding


class Transformer(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=params.d_model,
                                        max_len=params.max_len,
                                        cov_dim=params.cov_dim*2,
                                        drop_prob=params.dropout,
                                        device=params.device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=params.d_model,
                                                  ffn_hidden=params.ffn_hidden,
                                                  n_head=params.n_head,
                                                  drop_prob=params.dropout)
                                     for _ in range(params.n_layers)])

        self.relu = nn.ReLU()
        self.device = params.device

        # # 修改为两个输出头：一个输出均值，一个输出log方差
        # self.dense_mean = nn.Linear(params.d_model, params.cov_dim)
        # self.dense_logvar = nn.Linear(params.d_model, params.cov_dim)

        # 或者更简单的版本
        self.dense_mean = nn.Sequential(
            nn.Linear(params.d_model, params.d_model),
            nn.ReLU(),
            # nn.Dropout(params.dropout),
            nn.Linear(params.d_model, params.cov_dim)
        )

        self.dense_logvar = nn.Sequential(
            nn.Linear(params.d_model, params.d_model),
            nn.ReLU(),
            # nn.Dropout(params.dropout), 
            nn.Linear(params.d_model, params.cov_dim)
        )                



    def make_src_mask(self, src):
        # 存在非零则为True
        src_mask = (src != 0).any(dim=-1) .unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, x):
        src_mask = self.make_src_mask(x)
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        output = self.relu(x)

        
        mean = self.dense_mean(output)
        logvar = self.dense_logvar(output)


        aleatoric_std = torch.exp(0.5*logvar) 
        aleatoric_uncertainty = (torch.exp(logvar) ).mean(dim=-2)
        

        return {
            'mean': mean,          # 预测均值 f^W(x)
            'aleatoric_logvar': logvar,      # 对数方差 s_i = log(σ_i^2)
            'aleatoric_std': aleatoric_std,  # 异方差偶然不确定性 σ^2
            'aleatoric_uncertainty':aleatoric_uncertainty
        }
    

    def mc_predict(self, X, n_samples=50):
        """蒙特卡洛采样预测（返回均值和标准差）
        注意：此方法会临时将模型设为train模式以启用Dropout，
        即使是在torch.no_grad()环境下"""
        original_mode = self.training  # 保存原始状态
        self.train()  # ⚠️ 强制启用Dropout（关键步骤！）

        with torch.no_grad():
            # 采样多个预测
            means = []
            logvars = []

            for _ in range(n_samples):
                sample = self(X)
                means.append(sample['mean'])
                logvars.append(sample['aleatoric_logvar'])
            
            means = torch.stack(means)  # [n_samples, batch_size, seq_len]
            logvars = torch.stack(logvars)

            # 计算总不确定性
            total_mean = means.mean(dim=0) 
            aleatoric_logvar = logvars.mean(dim=0)          # 𝔼[log σ²] 均值

            aleatoric_variance = (torch.exp(logvars).mean(dim=0))  #数据不确定性的方差
            epistemic_variance = (means.var(dim=0))  #模型不确定性的方差
            total_variance = aleatoric_variance + epistemic_variance

            # 计算epistemic uncertainty（论文中的公式9）,对样本求平均
            aleatoric_uncertainty = (aleatoric_variance[:, :, :]) .mean(dim=-2) 
            epistemic_uncertainty = (epistemic_variance[:, :, :]) .mean(dim=-2)   # 沿序列维度平均


        self.train(original_mode)  # 恢复原始状态

        return {
            'mean': total_mean,
            'aleatoric_logvar':aleatoric_logvar,

            'total_std': total_variance.sqrt(),
            'epistemic_std': epistemic_variance.sqrt(),
            'aleatoric_std': aleatoric_variance.sqrt(),
            
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty
        }

    def kl_loss(self):
        """仅返回最后一层的KL散度"""
        return torch.tensor(0) 
    



