import pandas as pd
import pickle
import optuna
import implicit
import scipy.sparse as sp
import numpy as np
import json
import random

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_matrix(df, n_users, n_items):
    return sp.csr_matrix(
        (np.ones(len(df)), (df["user_id"].values, df["item_id"].values)),
        shape=(n_users, n_items)
    )


def hit_rate_at_k(model, user_item_mat, gt, sample_users, K=10):
    hits = 0
    for uid in sample_users:
        if uid not in gt or not gt[uid]:
            continue
        recs, _ = model.recommend(uid, user_item_mat[uid], N=K,
                                  filter_already_liked_items=True)
        if set(recs) & set(gt[uid]):
            hits += 1
    return hits / len(sample_users)


if __name__ == "__main__":
    print(">>> Loading data...")
    train_df = pd.read_csv("../data/splits/lastfm/train.csv")
    val_df   = pd.read_csv("../data/splits/lastfm/val.csv")
    test_df  = pd.read_csv("../data/splits/lastfm/test.csv")

    with open("../data/splits/lastfm/metadata.json") as f:
        meta = json.load(f)
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    print(f"    n_users={n_users}, n_items={n_items}")

    train_mat = build_matrix(train_df, n_users, n_items)

    val_gt = {}
    for uid, grp in val_df[val_df["label"] == 1].groupby("user_id"):
        val_gt[uid] = grp["item_id"].tolist()

    random.seed(42)
    val_sample = random.sample(list(val_gt.keys()), min(200, len(val_gt)))

    # -------------------------------------------------------------- #
    # Optuna: search on train_mat so val items are never filtered out #
    # -------------------------------------------------------------- #
    def objective(trial):
        params = {
            "factors":        trial.suggest_int("factors", 16, 512),
            "regularization": trial.suggest_float("regularization", 1e-3, 1.0, log=True),
            "iterations":     trial.suggest_int("iterations", 10, 100),
            "alpha":          trial.suggest_float("alpha", 1.0, 100.0),
        }
        np.random.seed(42)  # keep Optuna trials comparable
        model = implicit.als.AlternatingLeastSquares(
            factors=params["factors"],
            regularization=params["regularization"],
            iterations=params["iterations"],
            random_state=42,  # requires implicit >= 0.6; remove if older
            use_gpu=False,
        )
        model.fit(train_mat * params["alpha"])
        # Pass train_mat so filter_already_liked_items only masks training interactions,
        # leaving val items as valid candidates (consistent with final eval setup).
        return hit_rate_at_k(model, train_mat, val_gt, val_sample, K=10)

    print(">>> Running Optuna (50 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best = study.best_params
    print(f"    Best params: {best}")
    print(f"    Best val HR@10: {study.best_value:.4f}")

    # ------------------------------------------------------------------ #
    # Retrain on train + val, then generate test predictions (multi-seed) #
    # ------------------------------------------------------------------ #
    print(">>> Retraining on train+val with multiple seeds...")
    all_df  = pd.concat([train_df, val_df])
    all_mat = build_matrix(all_df, n_users, n_items)

    test_users = test_df["user_id"].unique().tolist()

    seeds = [42, 2024, 123]
    all_predictions = []

    for seed in seeds:
        print(f"\n>>> Training ALS with seed={seed}")
        np.random.seed(seed)  # fallback for implicit versions without random_state

        final_model = implicit.als.AlternatingLeastSquares(
            factors=best["factors"],
            regularization=best["regularization"],
            iterations=best["iterations"],
            random_state=seed,  # requires implicit >= 0.6; remove if older
            use_gpu=False,
        )
        final_model.fit(all_mat * best["alpha"])

        preds = {}
        for i, uid in enumerate(test_users):
            if i % 500 == 0:
                print(f"    {i}/{len(test_users)}")
            recs, _ = final_model.recommend(
                uid, all_mat[uid], N=10,
                filter_already_liked_items=True
            )
            preds[uid] = [int(r) for r in recs]

        all_predictions.append(preds)

    # Save multi-seed predictions
    with open("../predictions/lastfm/als_all_seeds.pkl", "wb") as f:
        pickle.dump(all_predictions, f)
    print("    Saved → predictions/lastfm/als_all_seeds.pkl")

    # Save single-seed prediction (first seed, seed=42)
    with open("../predictions/lastfm/als.pkl", "wb") as f:
        pickle.dump(all_predictions[0], f)
    print("    Saved → predictions/lastfm/als.pkl")

    # Save best hyper-parameters
    with open("../role_B/results/lastfm/als_best_params.pkl", "wb") as f:
        pickle.dump(best, f)
    print("    Saved → role_B/results/lastfm/als_best_params.pkl")

    print("    Done.")