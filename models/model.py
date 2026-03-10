import torch
import torch.nn as nn
import torch.nn.functional as F

class DualModalPretainModel(nn.Module):
    def __init__(self, input_channels, vocab_size=6, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        # --- 1. Raw Stream (物理结构对齐) ---
        # 核心设计：Kernel=3, Padding=1 完美对应 (t-1, t, t+1) 的符号逻辑
        self.raw_local_extractor = nn.Sequential(
            # Layer 1: 模拟符号提取窗口
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Layer 2: 稍微扩大感受野，加强特征
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        # Transformer 处理长时序列
        self.raw_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128), 
            num_layers=2
        )
        # 用 Raw Feature 预测符号 (Dense Prediction)
        self.raw_predictive_head = nn.Linear(d_model, vocab_size)

        # --- 2. Symbol Stream (Channel Fusion) ---
        self.sym_embedding = nn.Embedding(vocab_size, d_model, padding_idx=5)
        
        # 将 (C * D) 压缩为 D -> 融合多轴信息
        self.channel_fusion = nn.Sequential(
            nn.Linear(d_model * input_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.sym_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128), 
            num_layers=2
        )
        
        # Masked Symbol Prediction Head
        self.sym_pred_head = nn.Linear(d_model, vocab_size)
        
        # --- 3. Projectors (Contrastive) ---
        self.projector_raw = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 32))
        self.projector_sym = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 32))

    def forward(self, raw, sym):
        B, C, T = raw.shape
        
        # === Raw Stream ===
        r_emb = self.raw_local_extractor(raw).permute(0, 2, 1) # (B, T, D)
        r_seq = self.raw_transformer(r_emb)                    # (B, T, D)
        
        # TCC/Dense Output: Raw predicts Symbol Mode
        raw_pred_logits = self.raw_predictive_head(r_seq)      # (B, T, V)
        
        # Contrastive Rep
        r_pooled = r_seq.mean(dim=1)
        z_raw = F.normalize(self.projector_raw(r_pooled), dim=1)
        
        # === Symbol Stream (Fused) ===
        s_emb = self.sym_embedding(sym)
        # Permute for fusion: (B, T, C, D) -> Flatten C*D -> (B, T, C*D)
        s_emb = s_emb.permute(0, 2, 1, 3).reshape(B, T, C * self.d_model)
        
        # Fusion: (B, T, D)
        s_fused = self.channel_fusion(s_emb)
        s_seq = self.sym_transformer(s_fused)
        
        # Symbol Auto-regression (Masked Prediction)
        sym_pred_logits = self.sym_pred_head(s_seq)
        
        # Contrastive Rep
        s_pooled = s_seq.mean(dim=1)
        z_sym = F.normalize(self.projector_sym(s_pooled), dim=1)
        
        return z_raw, z_sym, sym_pred_logits, raw_pred_logits

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class FinetuneModel(nn.Module):
    def __init__(self, pretrained_model, num_classes, freeze_mode="partial"):
        super().__init__()
        
        # 1. 提取预训练结构
        self.raw_net = pretrained_model.raw_local_extractor
        self.raw_trans = pretrained_model.raw_transformer
        self.d_model = pretrained_model.d_model
        
        # 2. 详细冻结逻辑
        print(f">>> [Architecture] Fine-tune Mode: [{freeze_mode}]")
        
        # 先全部开启梯度
        for p in self.raw_net.parameters(): p.requires_grad = True
        for p in self.raw_trans.parameters(): p.requires_grad = True
            
        if freeze_mode == "all":
            for p in self.raw_net.parameters(): p.requires_grad = False
            for p in self.raw_trans.parameters(): p.requires_grad = False
        elif freeze_mode == "partial":
            for p in self.raw_net.parameters(): p.requires_grad = False
        elif freeze_mode == "cnn_tune":
            for p in self.raw_trans.parameters(): p.requires_grad = False

        # 3. 分类头优化
        # 增加 Dropout 是 Few-shot 微调的关键，防止从第 5 代开始死记硬背
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, raw, sym=None):
        # (B, C, T) -> CNN -> (B, D, T)
        x = self.raw_net(raw)
        
        # (B, D, T) -> Permute -> (B, T, D)
        x = x.permute(0, 2, 1)
        
        # Transformer 处理
        x = self.raw_trans(x) # (B, T, D)
        
        # --- 池化改进 ---
        # 相比 mean，在 Few-shot 中同时考虑 Max 和 Mean 往往更稳健
        # 这里我们先尝试 Mean，但加入 Dropout 保护
        x = x.mean(dim=1)  # (B, D)
        x = self.dropout(x)
        
        return self.classifier(x)

    def train(self, mode=True):
        """
        这个函数是解决训练下降的核心。
        它不仅设置 mode，还暴力覆盖了所有冻结层的 BN 行为。
        """
        super().train(mode)
        
        # 遍历所有子模块
        for m in self.modules():
            # 如果是 BatchNorm 且该模块的参数被冻结了
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                # 检查该模块下的第一个参数是否被冻结
                params = list(m.parameters())
                if len(params) > 0 and not params[0].requires_grad:
                    m.eval() # 强制设为 eval 模式，锁定 running_mean/var
                    m.track_running_stats = False # 彻底关闭统计跟踪
        
        # 如果 Transformer 被冻结，也要确保它处于 eval 模式 (主要是 LayerNorm 和 Dropout)
        if any(p.requires_grad == False for p in self.raw_trans.parameters()):
            self.raw_trans.eval()