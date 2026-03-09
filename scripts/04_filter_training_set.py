# ============================================
# step1_filter_training_set.py
# 基于之前的分类结果（training_set_similarity_analysis.py输出的saponin_analysis_robust.csv)筛选训练集，只保留HIGH_CONFIDENCE
# ============================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import pickle
import os
from pathlib import Path

# ============================================
# 1. 加载数据
# ============================================

print("\n" + "=" * 70)
print("STEP 1: FILTER TRAINING SET - KEEP ONLY HIGH_CONFIDENCE SAPONINS")
print("=" * 70)

base_path = 'D:\\LuYang\\CityU Study\\SDSC6002\\sanopin_project\\data\\'

# 加载训练集和分类结果
train_path = os.path.join(base_path, 'success_samples_train.csv')
analysis_path = os.path.join(base_path, 'saponin_analysis_robust.csv')  # 你的分析结果文件

train_df = pd.read_csv(train_path, engine='python')
analysis_df = pd.read_csv(analysis_path, engine='python')

print(f"📊 原始训练集: {len(train_df)} 个分子")
print(f"📊 分类结果: {len(analysis_df)} 个分子")

# ============================================
# 2. 合并并筛选HIGH_CONFIDENCE
# ============================================

# 确保长度一致
assert len(train_df) == len(analysis_df), "数据长度不匹配!"

# 合并
merged_df = train_df.copy()
merged_df['Classification'] = analysis_df['Classification']

# 统计各类别数量
print("\n📊 原始分类分布:")
class_counts = merged_df['Classification'].value_counts()
for cls, count in class_counts.items():
    print(f"  {cls}: {count} ({count / len(merged_df) * 100:.1f}%)")

# 筛选HIGH_CONFIDENCE
high_conf_df = merged_df[merged_df['Classification'] == 'HIGH_CONFIDENCE_SAPONIN'].copy()
print(f"\n🎯 筛选后: {len(high_conf_df)} 个HIGH_CONFIDENCE皂苷")

# ============================================
# 3. 生成并保存指纹
# ============================================

print("\n🔬 生成指纹...")

morgan_gen = GetMorganGenerator(radius=3, fpSize=2048)


def smiles_to_fingerprint(smiles):
    """Convert SMILES to fingerprint with robust parsing"""
    if not isinstance(smiles, str) or smiles == "":
        return None

    # Fix nitro groups
    smiles = smiles.replace('N(=O)O', '[N+](=O)[O-]')
    smiles = smiles.replace('N(=O)[O]', '[N+](=O)[O-]')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
        except:
            return None

    return morgan_gen.GetFingerprint(mol)


# 生成指纹
fingerprints = []
valid_indices = []
valid_smiles = []

for idx, row in high_conf_df.iterrows():
    fp = smiles_to_fingerprint(row['SMILES'])
    if fp is not None:
        fingerprints.append(fp)
        valid_indices.append(idx)
        valid_smiles.append(row['SMILES'])

print(f"✅ 成功生成 {len(fingerprints)} 个有效指纹")

# ============================================
# 4. 转换为numpy数组格式（用于GPU计算）
# ============================================

print("\n💾 转换为numpy数组...")

# 将RDKit指纹转换为numpy数组
fp_arrays = []
for fp in fingerprints:
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fp_arrays.append(arr)

fp_matrix = np.stack(fp_arrays)
print(f"✅ 指纹矩阵形状: {fp_matrix.shape}")

# ============================================
# 5. 分析内部相似度分布（用于确定阈值）
# ============================================

print("\n📊 分析HIGH_CONFIDENCE内部相似度分布...")

# 随机采样计算相似度（避免O(n²)）
n_samples = min(10000, len(fingerprints) * (len(fingerprints) - 1) // 2)
similarities = []

if len(fingerprints) > 100:
    # 随机选择100个分子计算两两相似度
    sample_size = min(100, len(fingerprints))
    sample_indices = np.random.choice(len(fingerprints), sample_size, replace=False)
    sample_fps = [fingerprints[i] for i in sample_indices]

    for i in range(len(sample_fps)):
        for j in range(i + 1, len(sample_fps)):
            sim = DataStructs.TanimotoSimilarity(sample_fps[i], sample_fps[j])
            similarities.append(sim)
else:
    # 小数据集直接计算全部
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(sim)

similarities = np.array(similarities)
print(f"\n📈 相似度统计:")
print(f"  均值: {np.mean(similarities):.4f}")
print(f"  中位数: {np.median(similarities):.4f}")
print(f"  标准差: {np.std(similarities):.4f}")
print(f"  最小值: {np.min(similarities):.4f}")
print(f"  最大值: {np.max(similarities):.4f}")
print(f"  下四分位数 (Q1): {np.percentile(similarities, 25):.4f}")
print(f"  上四分位数 (Q3): {np.percentile(similarities, 75):.4f}")

# 建议阈值
recommended_threshold = np.percentile(similarities, 25)  # 下四分位数
print(f"\n💡 建议阈值: {recommended_threshold:.4f} (基于下四分位数)")
print(f"  或使用固定阈值: 0.20")

# ============================================
# 6. 保存结果
# ============================================

print("\n💾 保存结果...")

output_dir = Path('D:\\LuYang\\CityU Study\\SDSC6002\\sanopin_project\\results\\training_set_filtered')
output_dir.mkdir(exist_ok=True)

# 保存筛选后的训练集信息
# high_conf_df.to_csv(output_dir / 'high_confidence_training_set.csv', index=False)
high_conf_df[['#Name', 'LogHD50', 'Co-solvent', 'SMILES', 'Reference']].to_csv(
    output_dir / 'high_confidence_training_set.csv', index=False
)
print(f"✅ 保存训练集: {output_dir}/high_confidence_training_set.csv")

# 保存指纹矩阵（numpy格式）
np.save(output_dir / 'high_confidence_fingerprints.npy', fp_matrix)
print(f"✅ 保存指纹矩阵: {output_dir}/high_confidence_fingerprints.npy")

# 保存SMILES列表
with open(output_dir / 'high_confidence_smiles.txt', 'w') as f:
    for smi in valid_smiles:
        f.write(smi + '\n')
print(f"✅ 保存SMILES: {output_dir}/high_confidence_smiles.txt")

# 保存相似度分布结果
with open(output_dir / 'similarity_distribution.txt', 'w') as f:
    f.write(f"Total molecules: {len(fingerprints)}\n")
    f.write(f"Mean similarity: {np.mean(similarities):.4f}\n")
    f.write(f"Median similarity: {np.median(similarities):.4f}\n")
    f.write(f"Std: {np.std(similarities):.4f}\n")
    f.write(f"Q1 (25%): {np.percentile(similarities, 25):.4f}\n")
    f.write(f"Q3 (75%): {np.percentile(similarities, 75):.4f}\n")
    f.write(f"Recommended threshold: {recommended_threshold:.4f}\n")

# 保存完整结果用于后续步骤
with open(output_dir / 'training_data.pkl', 'wb') as f:
    pickle.dump({
        'fingerprints': fingerprints,  # RDKit指纹对象
        'fp_matrix': fp_matrix,  # numpy数组
        'smiles': valid_smiles,
        'metadata': {
            'n_molecules': len(fingerprints),
            'mean_similarity': float(np.mean(similarities)),
            'recommended_threshold': float(recommended_threshold)
        }
    }, f)
print(f"✅ 保存完整数据: {output_dir}/training_data.pkl")

print("\n" + "=" * 70)
print("STEP 1 COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📁 结果保存在: {output_dir.absolute()}")
