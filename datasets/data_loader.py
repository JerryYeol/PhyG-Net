# import torch
# from torch.utils.data import Dataset, random_split
# import os
# from utils.utils import generate_symbol_indices  # 导入符号生成工具

# class DualModalDataset(Dataset):
#     def __init__(self, file_path):
#         self.data_loaded = True
#         try:
#             loaded = torch.load(file_path, map_location='cpu')
#             if isinstance(loaded, dict):
#                 self.samples = loaded.get('samples', loaded.get('data'))
#                 self.labels = loaded.get('labels', loaded.get('y'))
#             else:
#                 self.samples = loaded[0]; self.labels = loaded[1]
            
#             if not isinstance(self.samples, torch.Tensor): self.samples = torch.tensor(self.samples)
#             if not isinstance(self.labels, torch.Tensor): self.labels = torch.tensor(self.labels).long()
#             if self.samples.ndim == 3 and self.samples.shape[2] < self.samples.shape[1]:
#                 self.samples = self.samples.permute(0, 2, 1)
#         except Exception as e:
#             print(f"Data load error: {e}"); self.data_loaded = False

#     def __len__(self): return len(self.samples) if self.data_loaded else 0
#     def __getitem__(self, idx):
#         raw = self.samples[idx].float()
#         # 实时生成符号，确保逻辑一致
#         symbols = generate_symbol_indices(raw) 
#         label = self.labels[idx]
#         return raw, symbols, label

# def collate_fn_pad(batch):
#     raws, syms, labels = zip(*batch)
#     bs = len(batch)
#     max_len = max([r.shape[-1] for r in raws])
#     nc = raws[0].shape[0]
    
#     p_raw = torch.zeros(bs, nc, max_len)
#     p_sym = torch.full((bs, nc, max_len), 5) # 5 is PAD token
    
#     for i in range(bs):
#         curr = raws[i].shape[-1]
#         p_raw[i, :, :curr] = raws[i]
#         p_sym[i, :, :curr] = syms[i]
        
#     return p_raw, p_sym, torch.tensor(labels).long()

# def get_few_shot_datasets(data_dir, ratio=0.01):
#     train_path = os.path.join(data_dir, 'train.pt')
#     test_path = os.path.join(data_dir, 'test.pt')
#     val_path = os.path.join(data_dir, 'val.pt')
    
#     if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(val_path)):
#         raise ValueError("Missing data files (train, test, or val)")

#     # 1. 获取 Train 和 Test 的总数量
#     def get_len(path):
#         data = torch.load(path, map_location='cpu')
#         if isinstance(data, dict):
#             samples = data.get('samples', data.get('data'))
#         else:
#             samples = data[0]
#         return len(samples)

#     n_train = get_len(train_path)
#     n_test = get_len(test_path)
#     total_source = n_train + n_test
    
#     # 2. 计算目标微调样本数
#     target_k = int(total_source * ratio)
#     target_k = max(1, target_k)
    
#     # 3. 加载验证集
#     val_ds = DualModalDataset(val_path)
#     val_total = len(val_ds)
    
#     # 4. 边界处理
#     train_size = target_k
#     if train_size > val_total:
#         print(f"[Warning] Ratio ({ratio}) requries {train_size} samples, but Val only has {val_total}.")
#         training_size = val_total
#         test_size = 0
#     else:
#         training_size = train_size
#         test_size = val_total - training_size
        
#     print(f"\n[Few-Shot Calculation]")
#     print(f"   Baseline Total (Train+Test): {total_source}")
#     print(f"   Ratio: {ratio:.2%} -> Target N: {target_k}")
#     print(f"   Source Pool: Val Set ({val_total})")
#     print(f"   Final Split: Train({training_size}) + Test({test_size}) subset from Val")

#     # 5. 划分数据集
#     return random_split(val_ds, [training_size, test_size], generator=torch.Generator().manual_seed(42))

# 第二代
# import torch
# import numpy as np
# from torch.utils.data import Dataset, Subset
# import os
# # 假设 generate_symbol_indices 在 utils.utils 里，保持不变
# from utils.utils import generate_symbol_indices 

# class DualModalDataset(Dataset):
#     def __init__(self, file_path):
#         self.data_loaded = True
#         self.groups = None # 新增：用于存储病人ID
        
#         try:
#             loaded = torch.load(file_path, map_location='cpu')
#             if isinstance(loaded, dict):
#                 self.samples = loaded.get('samples', loaded.get('data'))
#                 self.labels = loaded.get('labels', loaded.get('y'))
#                 # 尝试加载病人ID信息
#                 self.groups = loaded.get('groups', loaded.get('subject_id', None))
#             else:
#                 # 兼容旧格式 (samples, labels)
#                 self.samples = loaded[0]
#                 self.labels = loaded[1]
            
#             # 转 Tensor
#             if not isinstance(self.samples, torch.Tensor): 
#                 self.samples = torch.tensor(self.samples)
#             if not isinstance(self.labels, torch.Tensor): 
#                 self.labels = torch.tensor(self.labels).long()
            
#             # 维度调整 (B, T, C) -> (B, C, T)
#             if self.samples.ndim == 3 and self.samples.shape[2] < self.samples.shape[1]:
#                 self.samples = self.samples.permute(0, 2, 1)
                
#             # 如果 groups 存在，转换为 numpy 数组以便索引
#             if self.groups is not None and not isinstance(self.groups, np.ndarray):
#                 self.groups = np.array(self.groups)
                
#         except Exception as e:
#             print(f"Data load error: {e}")
#             self.data_loaded = False

#     def __len__(self): 
#         return len(self.samples) if self.data_loaded else 0

#     def __getitem__(self, idx):
#         raw = self.samples[idx].float()
#         symbols = generate_symbol_indices(raw) # 实时生成符号
#         label = self.labels[idx]
#         return raw, symbols, label

# # collate_fn_pad 保持不变，可以直接复用你原来的
# def collate_fn_pad(batch):
#     raws, syms, labels = zip(*batch)
#     bs = len(batch)
#     max_len = max([r.shape[-1] for r in raws])
#     nc = raws[0].shape[0]
    
#     p_raw = torch.zeros(bs, nc, max_len)
#     p_sym = torch.full((bs, nc, max_len), 5) # 5 is PAD token
    
#     for i in range(bs):
#         curr = raws[i].shape[-1]
#         p_raw[i, :, :curr] = raws[i]
#         p_sym[i, :, :curr] = syms[i]
        
#     return p_raw, p_sym, torch.tensor(labels).long()

# # def get_few_shot_datasets(data_dir, ratio=0.01):
# #     """
# #     修正版：基于病人ID (Groups) 进行划分，彻底杜绝数据泄露。
# #     """
# #     train_path = os.path.join(data_dir, 'train.pt')
# #     test_path = os.path.join(data_dir, 'test.pt')
# #     val_path = os.path.join(data_dir, 'val.pt')
    
# #     if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(val_path)):
# #         raise ValueError("Missing data files (train, test, or val)")

# #     # 1. 计算目标总数量 (基于原始 train+test 的大小)
# #     def get_len(path):
# #         data = torch.load(path, map_location='cpu')
# #         if isinstance(data, dict):
# #             samples = data.get('samples', data.get('data'))
# #         else:
# #             samples = data[0]
# #         return len(samples)

# #     total_source = get_len(train_path) + get_len(test_path)
# #     target_k = int(total_source * ratio)
# #     target_k = max(1, target_k) # 至少选1个样本
    
# #     # 2. 加载 Val 数据集作为池子
# #     val_ds = DualModalDataset(val_path)
    
# #     # --- 核心修正逻辑开始 ---
    
# #     # 检查是否有 groups 信息
# #     if val_ds.groups is None:
# #         print("\n!!!!!!!!!! CRITICAL WARNING !!!!!!!!!!")
# #         print("数据集没有 'groups' (病人ID) 信息。无法进行按病人划分。")
# #         print("将回退到随机划分，但这会导致数据泄露，准确率将不可信！")
# #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
# #         # 回退逻辑 (不推荐，但防止报错)
# #         train_size = min(target_k, len(val_ds))
# #         test_size = len(val_ds) - train_size
# #         return torch.utils.data.random_split(val_ds, [train_size, test_size], 
# #                                              generator=torch.Generator().manual_seed(42))

# #     # 3. 按病人划分 (Group-level Split)
# #     unique_subjects = np.unique(val_ds.groups)
# #     np.random.seed(42) # 固定随机种子，保证可复现
# #     np.random.shuffle(unique_subjects) # 打乱病人顺序
    
# #     train_indices = []
# #     test_indices = []
    
# #     current_count = 0
# #     train_subjects = []
    
# #     # 贪心策略：逐个把病人放入训练集，直到样本数 >= target_k
# #     for subj in unique_subjects:
# #         # 找到属于该病人的所有样本索引
# #         subj_indices = np.where(val_ds.groups == subj)[0].tolist()
        
# #         if current_count < target_k:
# #             # 还没凑够，继续放进训练集
# #             train_indices.extend(subj_indices)
# #             current_count += len(subj_indices)
# #             train_subjects.append(subj)
# #         else:
# #             # 凑够了，剩下的病人全部放进测试集
# #             test_indices.extend(subj_indices)
            
# #     # 4. 构建 Subsets
# #     train_subset = Subset(val_ds, train_indices)
# #     test_subset = Subset(val_ds, test_indices)

# #     print(f"\n[Fixed Few-Shot Split - Group Level]")
# #     print(f"   Target Samples: ~{target_k} (Ratio: {ratio:.2%})")
# #     print(f"   Total Subjects in Val: {len(unique_subjects)}")
# #     print(f"   Selected Train Samples: {len(train_subset)} (from {len(train_subjects)} subjects)")
# #     print(f"   Selected Test Samples:  {len(test_subset)} (remaining subjects)")
# #     print(f"   Train Subjects: {train_subjects[:5]} ...") # 打印前几个看看
    
# #     return train_subset, test_subset

# import torch
# import numpy as np
# from torch.utils.data import Subset
# import os
# from collections import defaultdict

# def get_few_shot_datasets(data_dir, ratio=0.01, seed=42):
#     """
#     分层组划分：在保证不泄露病人的前提下，尽可能覆盖所有类别。
#     """
#     train_path = os.path.join(data_dir, 'train.pt')
#     val_path = os.path.join(data_dir, 'val.pt')
    
#     # 1. 计算目标样本数
#     def get_len(path):
#         data = torch.load(path, map_location='cpu')
#         samples = data.get('samples', data.get('data')) if isinstance(data, dict) else data[0]
#         return len(samples)

#     total_source = get_len(train_path) + get_len(os.path.join(data_dir, 'test.pt'))
#     target_k = max(1, int(total_source * ratio))
    
#     # 2. 加载 Val 数据集
#     val_ds = DualModalDataset(val_path)
#     if val_ds.groups is None:
#         raise ValueError("Group information missing! Cannot ensure non-leakage split.")

#     # 3. 统计每个病人的标签分布
#     # subject_to_labels: { 病人ID: [标签1, 标签2, ...] }
#     # subject_to_indices: { 病人ID: [样本索引1, 样本索引2, ...] }
#     subject_to_labels = defaultdict(list)
#     subject_to_indices = defaultdict(list)
    
#     for i in range(len(val_ds)):
#         subj = val_ds.groups[i]
#         label = int(val_ds.labels[i])
#         subject_to_labels[subj].append(label)
#         subject_to_indices[subj].append(i)

#     # 4. 确定每个病人的“代表性标签”（通常取众数或者第一个）
#     # 这样我们可以知道哪些病人代表哪些类
#     unique_subjects = list(subject_to_labels.keys())
#     subject_main_label = {s: max(set(labels), key=labels.count) for s, labels in subject_to_labels.items()}
    
#     # 5. 按类归类病人
#     # label_to_subjects: { 类别0: [病人A, 病人B], 类别1: [病人C] }
#     label_to_subjects = defaultdict(list)
#     for s, l in subject_main_label.items():
#         label_to_subjects[l].append(s)
    
#     all_classes = sorted(list(label_to_subjects.keys()))
    
#     # 6. 分层贪心选择
#     train_indices = []
#     selected_subjects = set()
    
#     np.random.seed(seed)
#     for cls in all_classes:
#         np.random.shuffle(label_to_subjects[cls])
    
#     # 第一轮：确保每个类至少有一个病人
#     for cls in all_classes:
#         if len(label_to_subjects[cls]) > 0:
#             subj = label_to_subjects[cls].pop(0)
#             train_indices.extend(subject_to_indices[subj])
#             selected_subjects.add(subj)
    
#     # 第二轮：如果样本数不够，继续从各个类中轮询添加病人，直到达到 target_k
#     # 这样能保证各类别比例大致均衡
#     curr_class_idx = 0
#     while len(train_indices) < target_k:
#         # 检查是否还有可用的病人
#         remaining_classes = [c for c in all_classes if len(label_to_subjects[c]) > 0]
#         if not remaining_classes:
#             break
            
#         cls = all_classes[curr_class_idx % len(all_classes)]
#         if len(label_to_subjects[cls]) > 0:
#             subj = label_to_subjects[cls].pop(0)
#             train_indices.extend(subject_to_indices[subj])
#             selected_subjects.add(subj)
        
#         curr_class_idx += 1

#     # 7. 剩下的所有病人进测试集
#     test_indices = []
#     for subj, indices in subject_to_indices.items():
#         if subj not in selected_subjects:
#             test_indices.extend(indices)

#     # 打印统计结果
#     train_labels = [int(val_ds.labels[i]) for i in train_indices]
#     unique_train_labels = np.unique(train_labels)
    
#     print(f"\n[Stratified Few-Shot Split]")
#     print(f"   Target Samples: ~{target_k}")
#     print(f"   Final Train Samples: {len(train_indices)} from {len(selected_subjects)} subjects")
#     print(f"   Classes Covered in Train: {unique_train_labels} ({len(unique_train_labels)}/{len(all_classes)})")
#     print(f"   Label Distribution in Train: {np.bincount(train_labels)}")
    
#     return Subset(val_ds, train_indices), Subset(val_ds, test_indices)

# 最新的
import torch
from torch.utils.data import Dataset, Subset
import os
import numpy as np
from collections import defaultdict
from utils.utils import generate_symbol_indices

class DualModalDataset(Dataset):
    def __init__(self, file_path):
        self.data_loaded = True
        try:
            # 尝试加载数据 (支持 dict 格式和 list 格式)
            loaded = torch.load(file_path, map_location='cpu')
            if isinstance(loaded, dict):
                self.samples = loaded.get('samples', loaded.get('data'))
                self.labels = loaded.get('labels', loaded.get('y'))
                # 关键：尝试读取 groups (患者ID)
                self.groups = loaded.get('groups') 
            else:
                self.samples = loaded[0]
                self.labels = loaded[1]
                self.groups = None
            
            # 类型转换
            if not isinstance(self.samples, torch.Tensor): 
                self.samples = torch.tensor(self.samples)
            if not isinstance(self.labels, torch.Tensor): 
                self.labels = torch.tensor(self.labels).long()
            
            # 维度调整：确保为 (Batch, Channel, Length)
            if self.samples.ndim == 3 and self.samples.shape[2] < self.samples.shape[1]:
                self.samples = self.samples.permute(0, 2, 1)
                
        except Exception as e:
            print(f"Data load error: {e}")
            self.data_loaded = False

    def __len__(self): 
        return len(self.samples) if self.data_loaded else 0

    def __getitem__(self, idx):
        raw = self.samples[idx].float()
        # 实时生成符号索引
        symbols = generate_symbol_indices(raw) 
        label = self.labels[idx]
        return raw, symbols, label

def collate_fn_pad(batch):
    """
    处理变长序列的 Padding 函数，确保 Batch 内长度一致
    """
    raws, syms, labels = zip(*batch)
    bs = len(batch)
    max_len = max([r.shape[-1] for r in raws])
    nc = raws[0].shape[0]
    
    p_raw = torch.zeros(bs, nc, max_len)
    p_sym = torch.full((bs, nc, max_len), 5) # 5 是预设的 PAD 符号
    
    for i in range(bs):
        curr = raws[i].shape[-1]
        p_raw[i, :, :curr] = raws[i]
        p_sym[i, :, :curr] = syms[i]
        
    return p_raw, p_sym, torch.tensor(labels).long()

def get_few_shot_datasets(data_dir, ratio=0.01):
    """
    保持原有接口，但实现受试者感知的 Few-shot 划分逻辑
    """
    train_path = os.path.join(data_dir, 'train.pt')
    test_path = os.path.join(data_dir, 'test.pt')
    val_path = os.path.join(data_dir, 'val.pt')
    
    if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(val_path)):
        raise ValueError("Missing data files (train, test, or val)")

    # 1. 计算基于原始训练+测试集的总样本数得出的目标 k
    def get_len(path):
        data = torch.load(path, map_location='cpu')
        return len(data.get('samples', data.get('data'))) if isinstance(data, dict) else len(data[0])

    total_base = get_len(train_path) + get_len(test_path)
    target_k = max(1, int(total_base * ratio))
    
    # 2. 加载 Val 作为采样池
    val_ds = DualModalDataset(val_path)
    val_total = len(val_ds)
    
    # 3. 跨受试者采样逻辑
    if hasattr(val_ds, 'groups') and val_ds.groups is not None:
        group_to_indices = defaultdict(list)
        # 将相同患者的样本索引归类
        for i in range(val_total):
            group_to_indices[val_ds.groups[i]].append(i)
            
        unique_groups = list(group_to_indices.keys())
        n_groups = len(unique_groups)
        
        # 计算每个患者应取的样本数 (尽量均匀)
        idxs_per_group = max(1, target_k // n_groups)
        
        train_indices = []
        rng = np.random.default_rng(42) # 固定种子
        
        for gid in unique_groups:
            choices = group_to_indices[gid]
            rng.shuffle(choices)
            train_indices.extend(choices[:idxs_per_group])
            
        # 补齐不足的部分或进行裁剪
        if len(train_indices) < target_k:
            remaining = [i for i in range(val_total) if i not in set(train_indices)]
            rng.shuffle(remaining)
            train_indices.extend(remaining[:(target_k - len(train_indices))])
        
        train_indices = train_indices[:target_k]
        
        # 验证集剩余部分作为评估
        all_indices = set(range(val_total))
        test_indices = list(all_indices - set(train_indices))
        
        # 获取最终覆盖的患者数
        num_subjects = len(np.unique(val_ds.groups[train_indices]))
    else:
        # Fallback: 无 Group 信息则回退随机
        print("[Warning] No 'groups' found, falling back to random split.")
        train_indices = np.random.choice(val_total, min(target_k, val_total), replace=False)
        test_indices = list(set(range(val_total)) - set(train_indices))
        num_subjects = "Unknown"

    print(f"\n[Stratified Few-Shot Split]")
    print(f"   Target Samples: ~{target_k}")
    print(f"   Final Train Samples: {len(train_indices)} from {num_subjects} subjects")

    return Subset(val_ds, train_indices), Subset(val_ds, test_indices)