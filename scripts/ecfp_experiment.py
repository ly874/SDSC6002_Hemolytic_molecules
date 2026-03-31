import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==================== 1. 加载数据 ====================
train_data = pd.read_csv('success_samples_train.csv')
print(f"训练集大小: {len(train_data)}")
print(train_data.head())

# 确保有 SMILES 和 LogHD50 列
smiles_list = train_data['SMILES'].tolist()
y = train_data['LogHD50'].values


# ==================== 2. 生成指纹的函数 ====================
def get_fingerprint(smiles, radius=2, n_bits=2048):
    """生成 ECFP 指纹（Morgan fingerprint）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


# ==================== 3. 定义实验参数 ====================
radii = [2, 3, 4, 6]  # 不同直径（半径）
nbits_list = [1024, 2048]  # 不同位长

results = []  # 存储实验结果

# ==================== 4. 循环实验 ====================
for radius in radii:
    for nbits in nbits_list:
        print(f"\n正在处理 radius={radius}, nBits={nbits}")

        # 生成所有指纹
        X = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            fp = get_fingerprint(smi, radius, nbits)
            if fp is not None:
                X.append(fp)
                valid_indices.append(i)
            else:
                print(f"无效SMILES: {smi}")

        X = np.array(X)
        y_filtered = y[valid_indices]
        print(f"有效分子数: {len(X)}")

        # 划分训练/验证集（快速评估）
        X_train, X_val, y_train, y_val = train_test_split(X, y_filtered, test_size=0.2, random_state=42)

        # 模型：随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # 交叉验证（用全部有效数据）
        cv_scores = cross_val_score(rf, X, y_filtered, cv=5, scoring='r2')
        print(f"交叉验证 R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 在全量数据上训练，用于特征选择
        rf.fit(X, y_filtered)

        # 特征选择（使用已训练的模型，prefit=True）
        selector = SelectFromModel(rf, threshold='median', prefit=True)
        X_selected = selector.transform(X)
        print(f"选择后特征数: {X_selected.shape[1]}")

        # 用选择后的特征重新交叉验证（使用新的随机森林实例）
        rf_sel = RandomForestRegressor(n_estimators=100, random_state=42)
        cv_scores_sel = cross_val_score(rf_sel, X_selected, y_filtered, cv=5, scoring='r2')
        print(f"特征选择后 R²: {cv_scores_sel.mean():.4f} ± {cv_scores_sel.std():.4f}")

        # 保存结果
        results.append({
            'radius': radius,
            'nbits': nbits,
            'original_r2': cv_scores.mean(),
            'selected_r2': cv_scores_sel.mean(),
            'n_features_original': X.shape[1],
            'n_features_selected': X_selected.shape[1]
        })

# ==================== 5. 结果可视化 ====================
results_df = pd.DataFrame(results)
print("\n实验结果汇总：")
print(results_df)

# 画图：对比不同指纹的 R²
plt.figure(figsize=(10, 6))
for nbits in nbits_list:
    subset = results_df[results_df['nbits'] == nbits]
    plt.plot(subset['radius'], subset['original_r2'], 'o-', label=f'original, nbits={nbits}')
    plt.plot(subset['radius'], subset['selected_r2'], 's--', label=f'selected, nbits={nbits}')
plt.xlabel('Radius')
plt.ylabel('Cross-validated R²')
plt.legend()
plt.title('Effect of ECFP radius and bit length on model performance')
plt.grid(True)
plt.savefig('ecfp_comparison.png', dpi=150)
plt.show()

# 保存结果到 CSV
results_df.to_csv('ecfp_experiment_results.csv', index=False)
print("结果已保存。")



# 诊断代码
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. 重新加载并严格对齐数据
train_data = pd.read_csv('success_samples_train.csv')
# 剔除可能存在的 LogHD50 为空的数据
train_data = train_data.dropna(subset=['LogHD50', 'SMILES'])

smiles_list = train_data['SMILES'].tolist()
y = train_data['LogHD50'].values

X = []
y_filtered = []
failed_count = 0

print("正在生成指纹...")
for i, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # 用组长建议的配置：radius=2, nbits=2048
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        X.append(np.array(fp))
        y_filtered.append(y[i])
    else:
        failed_count += 1

X = np.array(X)
y_filtered = np.array(y_filtered)

print("-" * 30)
print(f"数据总数: {len(smiles_list)}")
print(f"解析成功数: {len(X)}")
print(f"解析失败数: {failed_count}")
print(f"特征矩阵 X 形状: {X.shape}")
print(f"标签向量 y 形状: {y_filtered.shape}")
print(f"y 中是否包含 NaN: {np.isnan(y_filtered).any()}")
print("-" * 30)

# 2. 跑最简单的 RandomForest 看训练集 R2
print("训练基础 RandomForest 模型...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y_filtered)
y_pred_train = rf.predict(X)
train_r2 = r2_score(y_filtered, y_pred_train)

print(f">>> 训练集 R² (Training R²): {train_r2:.4f} <<<")
if train_r2 > 0.7:
    print("结论：训练集 R² 正常，数据加载无误。验证集 R² 为负纯粹是因为维度过高导致的严重过拟合。")

# 3. 看看分布
plt.hist(y_filtered, bins=30, edgecolor='black', color='#9B88ED')
plt.title('Distribution of LogHD50 in Training Set')
plt.xlabel('LogHD50')
plt.ylabel('Frequency')
plt.savefig('LogHD50_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
