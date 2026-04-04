
import json
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from catboost import CatBoostRegressor



RANDOM_SEED = 42
FP_RADIUS = 3
FP_BITS = 2048


# Load best parameters from Optuna


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

X_train = np.stack(df_train["SMILES"].apply(smiles_to_ecfp).values)
y_train = df_train["LogHD50"].values

print(f"Training samples: {X_train.shape[0]}")
print(f"Feature dimension: {X_train.shape[1]}")


# Train final model


print("\nTraining final CatBoost model on full training set...")

final_model = CatBoostRegressor(**best_params)

final_model.fit(
    X_train,
    y_train,
    early_stopping_rounds=200,
    use_best_model=True
)



final_model.save_model(
    "catboost_final_train_only.cbm"
)

