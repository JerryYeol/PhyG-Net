import torch
import os

class Config:
    # ================= 1. 路径配置 =================
    PROJECT_NAME = "SymTCC"
    DATA_DIR = "/data/Docker_Liutianhao2025/SymTCC/data/HAR" 
    
    CKPT_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # 自动创建目录
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 文件名
    TRAIN_FILE = "train.pt"
    TEST_FILE = "test.pt"
    VAL_FILE = "val.pt"
    
    CKPT_NAME = "symbol_aligned_HAR_ckpt.pth"
    CKPT_PATH = os.path.join(CKPT_DIR, CKPT_NAME)

    # ================= 2. 数据规格 (根据您的 .pt 文件) =================
    # 🔥 核心修改：Outcome 1-6 -> 映射为 0-5，所以是 6 类
    NUM_CLASSES = 6       
    
    # 您的数据 Shape 是 (64, 360)
    INPUT_CHANNELS = 9   
    SEQ_LEN = 128         

    # ================= 3. 模型超参数 =================
    D_MODEL = 64          # 嵌入维度
    
    # ⚠️ 注意：VOCAB_SIZE 通常指将连续信号转为离散符号的数量 (Quantization Codebook)
    # 如果您的模型是将电压值切分为 Bin，建议设为 256, 1024 或 4096。
    # 如果您之前的 '6' 是指分类数，请忽略此变量，直接用上面的 NUM_CLASSES。
    VOCAB_SIZE = 4     

    # ================= 4. 训练与硬件 =================
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 预训练 (Pretrain) - 无监督或自监督
    PRETRAIN_EPOCHS = 50
    PRETRAIN_BATCH_SIZE = 64
    PRETRAIN_LR = 1e-3
    
    # 微调 (Finetune) - 也就是分类任务
    FINETUNE_EPOCHS = 100
    FINETUNE_BATCH_SIZE = 32
    
    # 少样本学习比例 (1.5% 的数据用于微调，其余冻结或不用)
    RATIO = 0.105