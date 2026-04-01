import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ==================== 1. 加载训练集和测试集 ====================
def load_and_process(file_path):
    df = pd.read_csv(file_path)
    # 确保列名匹配（根据你上传的文件，列名包含 SMILES 和 LogHD50）
    df = df.dropna(subset=['LogHD50', 'SMILES'])
    return df

train_df = load_and_process('success_samples_train.csv')
test_df = load_and_process('success_samples_test.csv')

print(f"训练集样本数: {len(train_df)}")
print(f"测试集样本数: {len(test_df)}")

# ==================== 2. 指纹生成 (Radius=3, nBits=2048) ====================
def get_fps(smiles_list, radius=3, n_bits=2048):
    X, y_idx = [], []
    for i, sm in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            X.append(np.array(fp))
            y_idx.append(i)
    return np.array(X), y_idx

print("\n正在生成指纹...")
X_train, train_idx = get_fps(train_df['SMILES'].tolist())
y_train = train_df['LogHD50'].iloc[train_idx].values

X_test, test_idx = get_fps(test_df['SMILES'].tolist())
y_test = test_df['LogHD50'].iloc[test_idx].values

# ==================== 3. 训练模型 (不进行特征选择) ====================
print("正在训练全量随机森林模型...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ==================== 4. 评估结果 ====================
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("-" * 30)
print(f"【实验结果 - ECFP6-2048】")
print(f"训练集 R²: {r2_train:.4f}")
print(f"测试集 R²: {r2_test:.4f}")
print(f"测试集 MAE: {mae_test:.4f}")
print("-" * 30)

# ==================== 5. 可视化：预测 vs 实际 ====================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='#9B88ED', label=f'Test R² = {r2_test:.3f}')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual LogHD50')
plt.ylabel('Predicted LogHD50')
plt.title('Actual vs Predicted (ECFP6-2048)')
plt.legend()
plt.savefig('test_set_performance.png', dpi=300) # 显式保存图片
plt.show()