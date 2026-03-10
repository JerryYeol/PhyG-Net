import torch
import torch.nn.functional as F

def generate_symbol_indices(tensor_data):
    """ 
    (C, T) Float -> (C, T) Long Index (1-4) 
    使用平滑与去噪逻辑的更新版: Kernel=3逻辑对齐
    """
    x = tensor_data.float()
    
    # [3D 转换] 确保输入为 (Batch, C, T) 以支持池化和填充
    is_2d = (x.ndim == 2)
    if is_2d:
        x = x.unsqueeze(0) # (1, C, T)

    # 1. 平滑去噪 (Kernel=5)
    # padding=2 保证输出长度仍为 T
    x_smooth = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
    
    # 2. 注入微量噪声
    noise = torch.randn_like(x_smooth, device=x.device) * 1e-6
    x_smooth = x_smooth + noise
    
    # 3. 填充边界以计算 L 和 R (保持长度 T)
    padded = F.pad(x_smooth, (1, 1), mode='replicate') # (1, C, T+2)
    
    # 抽取邻域
    L = padded[..., :-2]      # t-1
    M = padded[..., 1:-1]     # t
    R = padded[..., 2:]       # t+1
    
    # 4. 符号分类逻辑 (A:1, V:2, U:3, D:4)
    mask_A = (M > L) & (M > R)                # 顶 (Peak)
    mask_V = (M < L) & (M < R)                # 底 (Valley)
    mask_U = (~mask_A) & (~mask_V) & (L < R)  # 上升 (Rise)
    
    # 初始化为 4 (Fall/Fall-through)
    indices = torch.full_like(M, 4, dtype=torch.long)
    indices[mask_U] = 3
    indices[mask_V] = 2
    indices[mask_A] = 1
    
    # [还原维度]
    if is_2d:
        indices = indices.squeeze(0) # Restore to (C, T)
    
    return indices.long()