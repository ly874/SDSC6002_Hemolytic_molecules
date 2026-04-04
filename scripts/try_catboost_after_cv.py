import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from catboost import CatBoostRegressor, Pool

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



RANDOM_SEED = 42
FP_RADIUS = 3          # ECFP6
FP_BITS = 2048         # full feature



with open("best_catboost_params.json", "r") as f:
    best_params = json.load(f)

best_params.update({
    "loss_function": "RMSE",
    "random_seed": RANDOM_SEED,
    "verbose": 200
})

print("Loaded best CatBoost parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")


# ECFP6 fingerprint generator


morgan_gen = GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_BITS
)

def smiles_to_ecfp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(FP_BITS, dtype=np.float32)
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)




df_train = pd.read_csv("../data/success_samples_train.csv")
df_test = pd.read_csv("../data/success_samples_test.csv")

print(f"Training set size: {len(df_train)}")
print(f"Test set size: {len(df_test)}")

X_train = np.stack(df_train["SMILES"].apply(smiles_to_ecfp).values)
y_train = df_train["LogHD50"].values

X_test = np.stack(df_test["SMILES"].apply(smiles_to_ecfp).values)
y_test = df_test["LogHD50"].values

train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)


# Train CatBoost


model = CatBoostRegressor(**best_params)

model.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=100,
    use_best_model=True
)


# Evaluate performance on test set


print("\nModel evaluation on test set:")

y_pred = model.predict(test_pool)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE : {mae:.4f}")
print(f"Test R²  : {r2:.4f}")



# Feature importance


feature_importance = model.get_feature_importance()
top_idx = np.argsort(feature_importance)[::-1][:5]

print("\nTop 5 important ECFP6 bit indices:")
print(top_idx)




# Visualization

sns.set_theme(style="whitegrid", font_scale=1.2)


plt.figure(figsize=(7, 7))
sns.scatterplot(
    x=y_test,
    y=y_pred,
    s=70,
    alpha=0.75,
    edgecolor="white"
)

min_val = min(y_test.min(), y_pred.min()) - 0.3
max_val = max(y_test.max(), y_pred.max()) + 0.3
plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

plt.xlabel("Experimental LogHD$_{50}$")
plt.ylabel("Predicted LogHD$_{50}$")
plt.title("CatBoost Prediction on Test Set")

textstr = (
    f"$R^2$ = {r2:.3f}\n"
    f"RMSE = {rmse:.3f}\n"
    f"MAE = {mae:.3f}"
)
plt.text(
    0.05, 0.95,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

plt.tight_layout()
plt.savefig("catboost_test_prediction_scatter.png", dpi=300)
plt.close()

residuals = y_test - y_pred

plt.figure(figsize=(7, 5))
sns.histplot(
    residuals,
    bins=25,
    kde=True,
    color="steelblue"
)

plt.axvline(0, color="k", linestyle="--", lw=1.5)
plt.xlabel("Residual (Experimental − Predicted)")
plt.title("Residual Distribution (Test Set)")

plt.tight_layout()
plt.savefig("catboost_test_residual_distribution.png", dpi=300)
plt.close()
