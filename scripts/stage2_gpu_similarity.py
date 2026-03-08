# stage2_gpu_similarity.py
# GPU加速版本 - 使用PyTorch + CUDA

import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm
import time
from google.colab import drive
import gc
import warnings

# ============================================
# 1. 挂载Google Drive
# ============================================
drive.mount('/content/drive')

# ============================================
# 2. 检查GPU
# ============================================
print("\n" + "=" * 50)
print("GPU Information")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    raise RuntimeError("CUDA not available! Please enable GPU in Colab: Runtime -> Change runtime type -> GPU")


# ============================================
# 3. 辅助函数
# ============================================
def smiles_to_fp_array(smiles_list, batch_size=10000):
    """
    将SMILES列表转换为指纹数组（CPU任务，分批处理避免内存溢出）
    """
    all_fps = []

    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Generating fingerprints"):
        batch_smiles = smiles_list[i:i + batch_size]
        batch_fps = []

        for smi in batch_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                batch_fps.append(arr)
            else:
                # 如果SMILES无效，用零向量代替
                warnings.warn(f"Invalid SMILES found: {smi[:50]}...", UserWarning)
                batch_fps.append(np.zeros(2048))

        all_fps.extend(batch_fps)

        # 每处理100万释放一次内存
        if (i // batch_size) % 10 == 0:
            gc.collect()

    return np.array(all_fps)


def tanimoto_gpu(batch_tensor, train_tensor, batch_size=5000):
    """
    GPU加速的批量Tanimoto相似度计算

    Args:
        batch_tensor: 候选分子指纹 [B, 2048]
        train_tensor: 训练集指纹 [624, 2048]
        batch_size: GPU内部批大小（防止显存溢出）

    Returns:
        avg_similarities: 平均相似度数组 [B]
    """
    all_avg_sims = []

    # 分批处理防止显存溢出
    for i in range(0, len(batch_tensor), batch_size):
        sub_batch = batch_tensor[i:i + batch_size]

        # 计算点积 [sub_batch, 624]
        dot = torch.mm(sub_batch, train_tensor.T)

        # 计算模长平方
        norm_sub = (sub_batch ** 2).sum(dim=1, keepdim=True)  # [sub_batch, 1]
        norm_train = (train_tensor ** 2).sum(dim=1)  # [624]

        # Tanimoto分母
        denominator = norm_sub + norm_train - dot
        denominator = torch.clamp(denominator, min=1e-8)

        # Tanimoto相似度矩阵 [sub_batch, 624]
        tanimoto = dot / denominator

        # 平均相似度 [sub_batch]
        avg_sim = tanimoto.mean(dim=1)

        all_avg_sims.append(avg_sim.cpu())

    return torch.cat(all_avg_sims).numpy()


# ============================================
# 4. 加载训练集并预计算到GPU
# ============================================
print("\n" + "=" * 50)
print("Loading Training Set")
print("=" * 50)

train_path = '/content/drive/MyDrive/SDSC6002/success_samples_train.csv'
train_df = pd.read_csv(train_path)
train_smiles = train_df['SMILES'].tolist()
print(f"Training samples: {len(train_smiles)}")

# 生成训练集指纹
print("\nGenerating training fingerprints...")
train_fps = smiles_to_fp_array(train_smiles)
print(f"Training fingerprints shape: {train_fps.shape}")

# 转换为PyTorch张量并移到GPU
print("\nMoving training fingerprints to GPU...")
train_tensor = torch.FloatTensor(train_fps).cuda()
print(f"Train tensor shape: {train_tensor.shape}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================
# 5. 设置路径和参数
# ============================================
print("\n" + "=" * 50)
print("Processing Candidate Molecules")
print("=" * 50)

# 路径设置
input_dir = '/content/drive/MyDrive/SDSC6002/outputs'  # 您的生成器输出目录
output_dir = '/content/drive/MyDrive/SDSC6002/saponin_results'  # 结果保存目录
os.makedirs(output_dir, exist_ok=True)

# 参数设置
threshold = 0.30  # 相似度阈值
gpu_batch_size = 10000  # GPU批大小（根据显存调整）
cpu_batch_size = 50000  # CPU批大小（读文件）

# 获取所有batch文件
batch_files = sorted([f for f in os.listdir(input_dir)
                      if f.startswith('batch_') and f.endswith('.txt')])
print(f"Found {len(batch_files)} batch files")

# ============================================
# 6. 主循环 - 分批处理
# ============================================
all_results = []
total_processed = 0
start_time = time.time()

for batch_idx, batch_file in enumerate(tqdm(batch_files, desc="Processing batches")):
    # 读取batch文件
    file_path = os.path.join(input_dir, batch_file)
    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]

    # 生成指纹（CPU）
    batch_fps = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            batch_fps.append(arr)
            valid_smiles.append(smi)

    if not batch_fps:
        continue

    # 转换为张量并移到GPU
    batch_tensor = torch.FloatTensor(np.array(batch_fps)).cuda()

    # GPU加速计算相似度
    avg_sims = tanimoto_gpu(batch_tensor, train_tensor, gpu_batch_size)

    # 筛选通过阈值的分子
    for smi, sim in zip(valid_smiles, avg_sims):
        if sim >= threshold:
            all_results.append({
                'SMILES': smi,
                'Similarity': float(sim),
                'Batch': batch_file
            })

    total_processed += len(smiles_list)

    # 清理GPU缓存
    del batch_tensor, avg_sims
    torch.cuda.empty_cache()
    gc.collect()

    # 每处理10个batch保存一次中间结果
    if (batch_idx + 1) % 10 == 0:
        temp_df = pd.DataFrame(all_results)
        temp_file = os.path.join(output_dir, f'interim_batch_{batch_idx + 1}.csv')
        temp_df.to_csv(temp_file, index=False)

        # 打印进度
        elapsed = time.time() - start_time
        rate = total_processed / elapsed
        print(f"\n--- Progress Report ---")
        print(f"Processed: {total_processed:,} molecules")
        print(f"Found in-AD: {len(all_results):,}")
        print(f"Speed: {rate:.0f} molecules/sec")
        print(f"Estimated remaining: {(len(batch_files) - batch_idx - 1) * len(smiles_list) / rate / 60:.1f} minutes")

# ============================================
# 7. 保存最终结果
# ============================================
print("\n" + "=" * 50)
print("Final Results")
print("=" * 50)

# 创建DataFrame
results_df = pd.DataFrame(all_results)
print(f"\nTotal molecules processed: {total_processed:,}")
print(f"Molecules in AD (≥{threshold}): {len(results_df):,}")
if len(results_df) > 0:
    print(f"Pass rate: {len(results_df) / total_processed * 100:.2f}%")
    print(f"Average similarity: {results_df['Similarity'].mean():.4f}")
    print(f"Max similarity: {results_df['Similarity'].max():.4f}")
    print(f"Min similarity: {results_df['Similarity'].min():.4f}")

# 按相似度排序
results_df = results_df.sort_values('Similarity', ascending=False)

# 保存完整结果
output_file = os.path.join(output_dir, 'all_in_ad_molecules.csv')
results_df.to_csv(output_file, index=False)
print(f"\n✅ Full results saved to: {output_file}")

# 保存按相似度分类的结果
if len(results_df) > 0:
    # 高相似度 (≥0.5)
    high_sim = results_df[results_df['Similarity'] >= 0.5]
    high_sim.to_csv(os.path.join(output_dir, 'high_similarity.csv'), index=False)
    print(f"High similarity (≥0.5): {len(high_sim)} molecules")

    # 中等相似度 (0.3-0.5)
    mid_sim = results_df[(results_df['Similarity'] >= 0.3) & (results_df['Similarity'] < 0.5)]
    mid_sim.to_csv(os.path.join(output_dir, 'medium_similarity.csv'), index=False)
    print(f"Medium similarity (0.3-0.5): {len(mid_sim)} molecules")

# 总运行时间
total_time = time.time() - start_time
print(f"\nTotal processing time: {total_time / 60:.2f} minutes")

print("\n" + "=" * 50)
print("✅ Stage 2 GPU Processing Complete!")
print("=" * 50)