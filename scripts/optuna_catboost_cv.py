
import json
import numpy as np
import pandas as pd
import optuna

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# Global configuration


RANDOM_SEED = 42
FP_RADIUS = 3        # ECFP6
FP_BITS = 2048       # 全特征
N_SPLITS = 5         # 5-fold CV
N_TRIALS = 50        # Optuna trials


# ECFP6 fingerprint generator


morgan_gen = GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_BITS
)

def smiles_to_ecfp(smiles: str) -> np.ndarray:
    """
    Convert SMILES to ECFP6 fingerprint.
    Invalid SMILES will return zero vector.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(FP_BITS, dtype=np.float32)
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)





df_train = pd.read_csv("../data/success_samples_train.csv")

X = np.stack(df_train["SMILES"].apply(smiles_to_ecfp).values)
y = df_train["LogHD50"].values

print(f"Training samples: {X.shape[0]}")
print(f"Fingerprint dimension: {X.shape[1]}")


# Optuna objective function


def objective(trial):
    """
    Optuna objective:
    - Sample CatBoost hyperparameters
    - Evaluate using 5-fold CV
    - Return mean RMSE
    """

    params = {
        # Model capacity
        "iterations": trial.suggest_int("iterations", 500, 2500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        ),

        # Regularization & randomness (small dataset critical)
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 1.0, 30.0, log=True
        ),
        "random_strength": trial.suggest_float(
            "random_strength", 0.1, 10.0, log=True
        ),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.0, 2.0
        ),

        # Fixed parameters
        "loss_function": "RMSE",
        "random_seed": RANDOM_SEED,
        "verbose": False
    }

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    return float(np.mean(rmse_scores))


# Run Optuna study


print("\nStarting Optuna hyperparameter optimization...")

study = optuna.create_study(
    direction="minimize",
    study_name="CatBoost_ECFP6_Optuna",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
)

study.optimize(objective, n_trials=N_TRIALS)


# Output results

print(f"Best CV RMSE: {study.best_value:.4f}")
print("Best hyperparameters:")

for key, value in study.best_params.items():
    print(f"  {key}: {value}")



with open("best_catboost_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
