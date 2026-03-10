import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os
from sklearn.metrics import f1_score
from tqdm import tqdm  # 导入 tqdm

# 导入我们的模块
from configs.config import Config
from utils.metrics import NTXentLoss
from datasets.data_loader import DualModalDataset, collate_fn_pad, get_few_shot_datasets
from models.model import DualModalPretainModel, FinetuneModel

# ==========================================
# Helper Functions (Train/Eval Loops)
# ==========================================

def run_common_train(model, dl_train, dl_test, device, epochs=100):
    """ 通用的训练/评估循环 """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return run_train_loop(model, dl_train, dl_test, optimizer, device, epochs)

def run_common_train_with_optim(model, dl_train, dl_test, optimizer, device, epochs):
    """ 接受自定义优化器的训练循环 """
    return run_train_loop(model, dl_train, dl_test, optimizer, device, epochs)

def run_train_loop(model, dl_train, dl_test, optimizer, device, epochs):
    crit = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_f1 = 0.0
    
    # 为 Epoch 添加进度条
    pbar_epoch = tqdm(range(epochs), desc="Training Epochs", unit="epoch")
    for epoch in pbar_epoch:
        # --- Train Loop ---
        model.train()
        train_loss = []
        # 为 Batch 训练添加进度条
        for raw, _, y in tqdm(dl_train, desc=f"Ep {epoch+1} Train", leave=False):
            raw, y = raw.to(device), y.to(device).long()
            optimizer.zero_grad()
            out = model(raw)
            loss = crit(out, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        # --- Eval Loop ---
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for raw, _, y in tqdm(dl_test, desc=f"Ep {epoch+1} Eval", leave=False):
                raw, y = raw.to(device), y.to(device).long()
                out = model(raw)
                preds = out.argmax(dim=1).cpu().numpy()
                targets = y.cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets)
        
        # --- Calculate Metrics ---
        val_preds = np.array(all_preds)
        val_targets = np.array(all_targets)
        
        acc = (val_preds == val_targets).mean()
        f1 = f1_score(val_targets, val_preds, average='weighted')
        
        # 更新最佳指标
        if acc > best_acc: 
            best_acc = acc
            best_f1 = f1
        
        # 在进度条末尾实时更新信息
        pbar_epoch.set_postfix({"Acc": f"{acc:.2%}", "Best": f"ACC:{best_acc:.2%} F1:{best_f1:.2%}", "F1": f"{f1:.4f}"})
        
        if (epoch+1) % 10 == 0:
            print(f"   Ep {epoch+1:03d} | Val Acc: {acc:.2%} (Best: {best_acc:.2%}) | F1: {f1:.4f}")
    
    return best_acc, best_f1

# ==========================================
# Task Procedures
# ==========================================

def run_pretrain(data_dir, save_path):
    print("\n" + "="*50 + "\n   MODE: Pre-training (Align Window + Channel Fusion)\n" + "="*50)
    
    # Init Data
    ds_train = DualModalDataset(os.path.join(data_dir, Config.TRAIN_FILE))
    ds_test = DualModalDataset(os.path.join(data_dir, Config.TEST_FILE)) 
    input_channels = ds_train.samples.shape[1]
    
    full_ds = ConcatDataset([ds_train, ds_test])
    dl = DataLoader(full_ds, batch_size=Config.PRETRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad, drop_last=True)
    
    # Init Model
    model = DualModalPretainModel(input_channels=input_channels, d_model=Config.D_MODEL).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.PRETRAIN_LR, weight_decay=1e-5)
    
    loss_con_fn = NTXentLoss(temperature=0.1)
    loss_ce_fn = nn.CrossEntropyLoss(ignore_index=5) 
    
    model.train()
    pbar_pretrain = tqdm(range(Config.PRETRAIN_EPOCHS), desc="Pretraining")
    for epoch in pbar_pretrain:
        loss_log = []
        acc_raw_sym = [] 
        
        for raw, sym, _ in tqdm(dl, desc=f"Ep {epoch+1} Batch", leave=False):
            if raw.shape[0] < 2: continue
            raw, sym = raw.to(Config.DEVICE), sym.to(Config.DEVICE).long()
            
            optimizer.zero_grad()
            
            # Forward
            z_raw, z_sym, sym_logits, raw_logits = model(raw, sym)
            
            # --- Targets (Channel Mode) ---
            sym_mode_val, _ = torch.mode(sym, dim=1) # (B, T)
            target_seq = sym_mode_val.view(-1)
            
            # Loss Calculation
            l_con = loss_con_fn(z_raw, z_sym)
            
            raw_seq_pred = raw_logits.view(-1, 6)
            l_dense = loss_ce_fn(raw_seq_pred, target_seq)
            
            sym_seq_pred = sym_logits.view(-1, 6)
            l_sym = loss_ce_fn(sym_seq_pred, target_seq)
            
            loss = l_con + 2.0 * l_dense + 0.5 * l_sym
            
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())
            
            valid_mask = (target_seq != 5)
            if valid_mask.sum() > 0:
                acc = (raw_seq_pred[valid_mask].argmax(1) == target_seq[valid_mask]).float().mean()
                acc_raw_sym.append(acc.item())
        
        pbar_pretrain.set_postfix({"Loss": f"{np.mean(loss_log):.4f}", "Raw->Sym Acc": f"{np.mean(acc_raw_sym):.2%}"})
        
    torch.save(model.state_dict(), save_path)
    print(f">>> Model saved to {save_path}")

def run_finetune(data_dir, load_path, ratio, epochs):
    print("\n" + "="*50 + "\n   MODE: Fine-tuning (Partial/Adaptive)\n" + "="*50)
    
    # 增加少样本数据集获取的提示
    print(f"Creating few-shot datasets (ratio={ratio})...")
    ft_train, ft_test = get_few_shot_datasets(data_dir, ratio=ratio)
    
    dl_train = DataLoader(ft_train, batch_size=Config.FINETUNE_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
    dl_test = DataLoader(ft_test, batch_size=Config.FINETUNE_BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad)
    
    input_channels = ft_train.dataset.samples.shape[1]
    
    # 🔥🔥🔥 核心修复：不要动态计算，直接从 Config 用于固定 6 类 🔥🔥🔥
    # 原始错误代码: num_classes = int(ys.max().item() + 1)
    num_classes = Config.NUM_CLASSES
    print(f">>> Target Classes: {num_classes}")

    # Load Pretrained
    base_model = DualModalPretainModel(input_channels=input_channels, d_model=Config.D_MODEL).to(Config.DEVICE)
    if os.path.exists(load_path):
        base_model.load_state_dict(torch.load(load_path, map_location=Config.DEVICE))
        print(f">>> Loaded Weights: {load_path}")
    else:
        print(">>> WARNING: Checkpoint not found, utilizing random init.")

    # Apply Partial Freezing Strategy
    model = FinetuneModel(base_model, num_classes, freeze_mode="partial").to(Config.DEVICE)
    
    # 差分学习率
    params = [
        {'params': model.classifier.parameters(), 'lr': 1e-3},
        {'params': filter(lambda p: p.requires_grad, model.raw_trans.parameters()), 'lr': 1e-4},
    ]
    
    optimizer = optim.Adam(params)
    
    print(f">>> Finetuning on {len(ft_train)} samples with Differential LR...")
    return run_common_train_with_optim(model, dl_train, dl_test, optimizer, Config.DEVICE, epochs)

def run_baseline(data_dir, ratio):
    print("\n" + "="*50 + "\n   MODE: Supervised Baseline (Train from Scratch)\n" + "="*50)
    
    print(f"Creating few-shot datasets (ratio={ratio})...")
    ft_train, ft_test = get_few_shot_datasets(data_dir, ratio=ratio)
    dl_train = DataLoader(ft_train, batch_size=Config.FINETUNE_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
    dl_test = DataLoader(ft_test, batch_size=Config.FINETUNE_BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad)
    
    input_channels = ft_train.dataset.samples.shape[1]
    temp_dl = DataLoader(ft_train, batch_size=100, shuffle=True, collate_fn=collate_fn_pad)
    _, _, ys = next(iter(temp_dl))
    num_classes = int(ys.max().item() + 1)
    
    # Model Setup (Random Init)
    base_model = DualModalPretainModel(input_channels=input_channels, d_model=Config.D_MODEL).to(Config.DEVICE)
    
    # 全参数训练 (freeze_encoder=False)
    model = FinetuneModel(base_model, num_classes, freeze_encoder=False).to(Config.DEVICE)
    
    print(f">>> Training from scratch on {len(ft_train)} samples...")
    return run_common_train(model, dl_train, dl_test, Config.DEVICE, epochs=Config.FINETUNE_EPOCHS)

# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    # 打印全局配置信息
    print(f"Project: {Config.PROJECT_NAME}")
    print(f"Device: {Config.DEVICE}")
    print(f"DataDir: {Config.DATA_DIR}")
    
    # 1. 预训练
    # run_pretrain(Config.DATA_DIR, save_path=Config.CKPT_PATH)
    
    # 2. 微调 (Round 1)
    acc1, f1_1 = run_finetune(Config.DATA_DIR, load_path=Config.CKPT_PATH, ratio=Config.RATIO, epochs=Config.FINETUNE_EPOCHS)
    
    # # 3. 再跑一次预训练 (保留原始逻辑，尽管通常不需要跑两次，但我完全保留您的执行流)
    # run_pretrain(Config.DATA_DIR, save_path=Config.CKPT_PATH)
    
    # # 4. 微调 (Round 2)
    # acc2, f1_2 = run_finetune(Config.DATA_DIR, load_path=Config.CKPT_PATH, ratio=Config.RATIO, epochs=Config.FINETUNE_EPOCHS)
    
    # 5. Baseline
    # baseline_acc, baseline_f1 = run_baseline(Config.DATA_DIR, ratio=Config.RATIO)
    
    print(f"\nFinal Results:")
    print(f"Finetune 1: {acc1:.2%}")
    # print(f"Finetune 2: {acc2:.2%}")
    # print(f"Baseline  : {baseline_acc:.2%}")