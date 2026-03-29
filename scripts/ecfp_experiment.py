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

        # 划分训练/验证集（可选，用于快速评估）
        X_train, X_val, y_train, y_val = train_test_split(X, y_filtered, test_size=0.2, random_state=42)

        # 模型：随机森林（你也可以换成其他模型）
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