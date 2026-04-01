#主要改动点：
#    1. 特征融合 (Feature Fusion)：将 2048 位 ECFP6 拓扑指纹与 5 个关键宏观理化描述符（LogP, MW, TPSA, HBD, HBA）拼接。
#      - 引入 LogP 以捕捉分子亲脂性对溶血活性的影响。
#    2. 模型正则化 (Model Regularization)：
#       - 增加随机森林迭代次数 (n_estimators=300) 以提升预测稳定性。
#      - 限制树深 (max_depth=20) 并设置叶节点最小样本数 (min_samples_leaf=2) 以抑制过拟合。

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors  # 新增：用于计算理化描述符
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


# ==================== 1. 加载训练集和测试集 ====================
def load_and_process(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['LogHD50', 'SMILES'])
    return df


train_df = load_and_process('success_samples_train.csv')
test_df = load_and_process('success_samples_test.csv')

print(f"训练集样本数: {len(train_df)}")
print(f"测试集样本数: {len(test_df)}")


# ==================== 2. 核心突破：指纹 + 理化描述符融合 ====================
def get_combined_features(smiles_list, radius=3, n_bits=2048):
    X, y_idx = [], []
    for i, sm in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            # A. 获取 ECFP6 指纹 (将其转为普通列表)
            fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))

            # B. 计算 5 个核心宏观理化性质
            try:
                mw = Descriptors.MolWt(mol)  # 分子量
                logp = Descriptors.MolLogP(mol)  # 脂水分配系数
                tpsa = Descriptors.TPSA(mol)  # 极性表面积
                hbd = Descriptors.NumHDonors(mol)  # 氢键供体数
                hba = Descriptors.NumHAcceptors(mol)  # 氢键受体数
            except:
                mw, logp, tpsa, hbd, hba = 0, 0, 0, 0, 0

            # C. 拼接特征：2048 (指纹) + 5 (理化) = 2053 维
            combined_features = fp + [mw, logp, tpsa, hbd, hba]
            X.append(np.array(combined_features))
            y_idx.append(i)

    return np.array(X), y_idx


print("\n正在生成融合特征 (ECFP6 + 5D Descriptors)...")
X_train, train_idx = get_combined_features(train_df['SMILES'].tolist())
y_train = train_df['LogHD50'].iloc[train_idx].values

X_test, test_idx = get_combined_features(test_df['SMILES'].tolist())
y_test = test_df['LogHD50'].iloc[test_idx].values

# ==================== 3. 训练升级版模型 ====================
print("正在训练升级版 Random Forest 模型...")
rf = RandomForestRegressor(
    n_estimators=300,  # 树的数量增加到 300，提升稳定性
    max_depth=20,  # 限制树深
    min_samples_leaf=2,  # 叶子节点至少包含2个样本，增强泛化能力
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ==================== 4. 评估结果 ====================
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("-" * 40)
print(f"【突破版实验结果 - ECFP6 + 宏观理化特征】")
print(f"特征总维度: {X_train.shape[1]}")
print(f"训练集 R²: {r2_train:.4f}")
print(f"测试集 R²: {r2_test:.4f}")
print(f"测试集 MAE: {mae_test:.4f}")
print("-" * 40)

# ==================== 5. 可视化：预测 vs 实际 ====================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='#FF6B6B', label=f'Test R² = {r2_test:.3f}')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual LogHD50')
plt.ylabel('Predicted LogHD50')
plt.title('Actual vs Predicted (ECFP6 + Descriptors)')
plt.legend()
plt.savefig('test_set_performance_upgraded.png', dpi=300, bbox_inches='tight')
plt.show()