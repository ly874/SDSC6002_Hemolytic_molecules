import pandas as pd
import numpy as np
import optuna
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==========================================
# 1. 提取 ECFP 指纹 (特征工程 - 导师要求 1)
# ==========================================
# 使用新版 API，避免刷屏警告。配置为 ECFP8 (radius=4) 和 4096 位以避免哈希碰撞
morgan_gen = GetMorganGenerator(radius=4, fpSize=4096)


def smiles_to_ecfp(smiles):
    """提取分子指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((4096,))
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp)


print("正在加载数据并提取分子指纹 (ECFP8, 4096 bits)...")

# 读取指定的训练集和测试集文件
df_train = pd.read_csv('../data/success_samples_train.csv')
df_test = pd.read_csv('../data/success_samples_test.csv')

print(f"训练集样本数: {len(df_train)}")
print(f"测试集样本数: {len(df_test)}")

# 将 SMILES 转换为指纹特征 (X)，并提取目标值 (y)
X_train = np.stack(df_train['SMILES'].apply(lambda x: smiles_to_ecfp(str(x))).values)
y_train = df_train['LogHD50'].values

X_test = np.stack(df_test['SMILES'].apply(lambda x: smiles_to_ecfp(str(x))).values)
y_test = df_test['LogHD50'].values

# 使用 Pool 包装数据，这是 CatBoost 推荐的高效数据格式
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

# ==========================================
# 2. 进阶版 Optuna 深度自动调参
# ==========================================
print("\n启动 Optuna 深度超参数搜索 (引入高级控制参数)...")


def objective(trial):
    """进阶版目标函数：包含更广阔、更深度的搜索空间"""
    param = {
        # 1. 基础结构参数
        'iterations': trial.suggest_int('iterations', 800, 3000),  # 允许建更多的树
        'depth': trial.suggest_int('depth', 4, 10),  # 允许树长得更深 (极度拟合能力)
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),  # 更细腻的学习率

        # 2. 正则化与抗过拟合参数 (极其重要)
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 25.0),  # L2正则化放宽上限
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),  # 采样随机性
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),  # 评分随机性

        # 3. 底层离散化参数
        'border_count': trial.suggest_categorical('border_count', [128, 254, 512]),

        # 固定参数
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }

    model = CatBoostRegressor(**param)

    # 增加 early_stopping 的耐心值，配合较小的学习率
    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=100,  # 从 50 增加到 100，给模型更多试错空间
        verbose=False
    )

    preds = model.predict(test_pool)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse


# 创建学习任务
study = optuna.create_study(direction='minimize', study_name="CatBoost_Advanced_Tuning")

# 【注意】因为搜索空间变大了，我们将尝试次数提升到 50 次 (大约需要几分钟)
study.optimize(objective, n_trials=50)

print("\n" + "=" * 40)
print("Optuna 深度调参完成！")
print(f"找到的终极最佳参数: {study.best_params}")
print(f"最佳验证集 RMSE: {study.best_value:.4f}")
print("=" * 40)

# ==========================================
# 3. 训练终极模型并评估
# ==========================================
print("\n正在使用终极参数训练最后一次模型...")

final_params = study.best_params
final_params.update({
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 200
})

best_model = CatBoostRegressor(**final_params)
best_model.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=100,
    use_best_model=True
)

# 最终评估
print("\n🔥 终极模型评估结果：")
y_pred = best_model.predict(test_pool)

final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_mae = mean_absolute_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"测试集 RMSE (均方根误差): {final_rmse:.4f}")
print(f"测试集 MAE  (平均绝对误差): {final_mae:.4f}")
print(f"测试集 R^2  (决定系数):     {final_r2:.4f}")
print("-" * 30)

# (如果你想继续画图，把第4步画图的代码接在这里即可)

# 查看特征重要性
feature_importances = best_model.get_feature_importance()
top_features_idx = np.argsort(feature_importances)[::-1][:5]
print(f"\n对预测溶血活性贡献最大的 5 个 ECFP8 指纹位点索引: {top_features_idx}")



import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 4. 绘制并保存“论文出版级”预测散点图
# ==========================================
print("\n正在生成并保存高质量散点图...")

# 设置 Seaborn 主题和全局高清晰度 (300 DPI 达到期刊出版要求)
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 创建一个 8x8 英寸的纯白底正方形画布
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制散点 (真实值 vs 预测值)
sns.scatterplot(
    x=y_test,
    y=y_pred,
    alpha=0.7,               # 半透明，点重叠时更有质感
    color="#1f77b4",         # 经典的科研蓝
    edgecolor="white",       # 增加白边，显得立体
    s=80,                    # 点的大小
    ax=ax
)

# 动态计算坐标轴的边界，确保正方形图的比例完美
min_val = min(min(y_test), min(y_pred)) - 0.5
max_val = max(max(y_test), max(y_pred)) + 0.5

# 画出理想预测情况下的对角线 (y = x)
ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    'k--',                   # 黑色虚线
    lw=2,                    # 线宽
    alpha=0.6,
    label="Ideal (y=x)"
)

# 在左上角添加包含三大评估指标的文本框
textstr = '\n'.join((
    f'$R^2$ = {final_r2:.4f}',
    f'RMSE = {final_rmse:.4f}',
    f'MAE = {final_mae:.4f}'
))
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
ax.text(
    0.05, 0.95, textstr,
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=props
)

# 设置精美的坐标轴标签和标题
ax.set_xlabel("Experimental LogHD$_{50}$", fontsize=15, fontweight='bold')
ax.set_ylabel("Predicted LogHD$_{50}$", fontsize=15, fontweight='bold')
ax.set_title("CatBoost Prediction of Hemolytic Activity", fontsize=16, fontweight='bold', pad=15)

# 限制坐标轴范围并确保横纵坐标刻度一致
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_aspect('equal', adjustable='box')  # 强制正方形

# 紧凑布局并保存高清图片
plt.tight_layout()
file_name = "LogHD50_Prediction_CatBoost.png"
plt.savefig(file_name, format='png', bbox_inches='tight')
print(f"图表已成功保存为当前目录下的: {file_name}")

# 在运行窗口显示图片
plt.show()