# Run on Colab
# Post-calculation analysis (histogram)
# 绘制相似度直方图
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from tqdm import tqdm

# ============================================
# 1. 加载所有分数（随机抽样）
# ============================================
scores_dir = '/content/drive/MyDrive/SDSC6002/saponin_results/similarity_scores'
score_files = sorted(glob.glob(scores_dir + '/scores_batch_*.npy'))
print(f"Found {len(score_files)} score files")

# 打乱文件顺序
random.seed(42)  # 固定种子，结果可重复
shuffled_files = score_files.copy()
random.shuffle(shuffled_files)

# 取前10%的文件
sample_ratio = 0.1
n_sample_files = int(len(shuffled_files) * sample_ratio)
sample_files = shuffled_files[:n_sample_files]
print(f"Sampling {len(sample_files)} files ({sample_ratio*100:.0f}%)")

# 从每个文件中抽样
sample_scores = []
for f in tqdm(sample_files, desc="Loading samples"):
    scores = np.load(f)
    # 每个文件取1%的分子
    n_sample = max(1, int(len(scores) * 0.01))
    sample_idx = random.sample(range(len(scores)), n_sample)
    sample_scores.extend(scores[sample_idx])

sample_scores = np.array(sample_scores)
print(f"Sampled {len(sample_scores):,} scores (≈{len(sample_scores)/44e6*100:.2f}% of total)")

# ============================================
# 2. 计算统计量
# ============================================
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
perc_values = np.percentile(sample_scores, percentiles)

print("\n📊 Similarity Distribution Statistics:")
for p, v in zip(percentiles, perc_values):
    print(f"  {p}th percentile: {v:.4f}")
print(f"  Max: {sample_scores.max():.4f}")
print(f"  Min: {sample_scores.min():.4f}")
print(f"  Mean: {sample_scores.mean():.4f}")
print(f"  Std: {sample_scores.std():.4f}")

# ============================================
# 3. 画直方图
# ============================================
plt.figure(figsize=(14, 8))

# 主直方图
plt.subplot(2, 1, 1)
counts, bins, patches = plt.hist(sample_scores, bins=100, alpha=0.7, 
                                  color='steelblue', edgecolor='black', linewidth=0.5)

# 标记关键阈值
thresholds = [0.15, 0.16, 0.17, 0.18, 0.20]
colors = ['green', 'lime', 'orange', 'red', 'darkred']
for th, col in zip(thresholds, colors):
    plt.axvline(x=th, color=col, linestyle='--', linewidth=2, 
                label=f'Threshold {th}')

plt.xlabel('Tanimoto Similarity', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Similarity Distribution: Virtual Library vs HIGH_CONFIDENCE Saponins', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 累积分布
plt.subplot(2, 1, 2)
sorted_scores = np.sort(sample_scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100

plt.plot(sorted_scores, cumulative, 'b-', linewidth=2)
for th, col in zip(thresholds, colors):
    plt.axvline(x=th, color=col, linestyle='--', linewidth=1, alpha=0.7)
    idx = np.searchsorted(sorted_scores, th)
    if idx < len(sorted_scores):
        pct = cumulative[idx]
        plt.plot(th, pct, 'o', color=col, markersize=8)
        plt.text(th+0.001, pct+2, f'{pct:.1f}%', fontsize=9, color=col)

plt.xlabel('Tanimoto Similarity', fontsize=12)
plt.ylabel('Cumulative Percentage (%)', fontsize=12)
plt.title('Cumulative Distribution', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/SDSC6002/saponin_results/similarity_distribution.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Histogram saved to: similarity_distribution.png")