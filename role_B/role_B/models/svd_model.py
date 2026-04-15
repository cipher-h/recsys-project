import pandas as pd
import pickle
import optuna
import random
from surprise import Dataset, Reader, SVD

optuna.logging.set_verbosity(optuna.logging.WARNING)


def hit_rate(preds, gt):
    hits, total = 0, 0
    for uid, items in preds.items():
        if uid not in gt or not gt[uid]:
            continue
        hits += int(len(set(items) & set(gt[uid])) > 0)
        total += 1
    return hits / total if total > 0 else 0


def get_top_k(model, users, all_items, user_seen, K=10):
    preds = {}
    for uid in users:
        seen = user_seen.get(uid, set())
        candidates = [i for i in all_items if i not in seen]
        scores = [(iid, model.predict(uid, iid).est)
                  for iid in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        preds[uid] = [int(iid) for iid, _ in scores[:K]]
    return preds


if __name__ == "__main__":
    print(">>> Loading data...")
    train_df = pd.read_csv("../data/splits/movielens-1m/train.csv")
    val_df   = pd.read_csv("../data/splits/movielens-1m/val.csv")
    test_df  = pd.read_csv("../data/splits/movielens-1m/test.csv")

    all_items = pd.concat([train_df, val_df, test_df])["item_id"].unique().tolist()

    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]], reader
    ).build_full_trainset()

    user_seen = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    val_gt = {}
    for uid, grp in val_df[val_df["label"] == 1].groupby("user_id"):
        val_gt[uid] = grp["item_id"].tolist()

    random.seed(42)
    val_sample = random.sample(list(val_gt.keys()), min(300, len(val_gt)))
    val_gt_sample = {u: val_gt[u] for u in val_sample}

    def objective(trial):
        params = {
            "n_factors": trial.suggest_int("n_factors", 20, 300),
            "n_epochs":  trial.suggest_int("n_epochs", 10, 60),
            "lr_all":    trial.suggest_float("lr_all", 1e-3, 0.1, log=True),
            "reg_all":   trial.suggest_float("reg_all", 0.01, 0.5, log=True),
        }
        model = SVD(**params)
        model.fit(train_data)
        preds = get_top_k(model, val_sample, all_items, user_seen, K=10)
        return hit_rate(preds, val_gt_sample)

    print(">>> Running Optuna (30 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best = study.best_params
    print(f"    Best params: {best}")
    print(f"    Best val HR@10: {study.best_value:.4f}")

    # ------------------------------------------------------------------ #
    # Retrain on train + val, then generate test predictions (multi-seed) #
    # ------------------------------------------------------------------ #
    print(">>> Retraining on train+val...")
    all_df = pd.concat([train_df, val_df])
    full_data = Dataset.load_from_df(
        all_df[["user_id", "item_id", "rating"]], reader
    ).build_full_trainset()

    test_users = test_df["user_id"].unique().tolist()
    user_seen_all = all_df.groupby("user_id")["item_id"].apply(set).to_dict()

    seeds = [42, 2024, 123]
    all_predictions = []

    for seed in seeds:
        print(f"\n>>> Training with seed={seed}")

        model = SVD(**best, random_state=seed)
        model.fit(full_data)

        preds = {}
        for i, uid in enumerate(test_users):
            if i % 500 == 0:
                print(f"    {i}/{len(test_users)}")

            seen = user_seen_all.get(uid, set())
            candidates = [item for item in all_items if item not in seen]

            scores = [(iid, model.predict(uid, iid).est)
                      for iid in candidates]
            scores.sort(key=lambda x: x[1], reverse=True)

            preds[uid] = [int(iid) for iid, _ in scores[:10]]

        all_predictions.append(preds)

    # Save multi-seed predictions
    with open("../predictions/movielens-1m/svd_all_seeds.pkl", "wb") as f:
        pickle.dump(all_predictions, f)
    print("    Saved → predictions/movielens-1m/svd_all_seeds.pkl")

    # Save single-seed prediction (first seed, seed=42)
    with open("../predictions/movielens-1m/svd.pkl", "wb") as f:
        pickle.dump(all_predictions[0], f)
    print("    Saved → predictions/movielens-1m/svd.pkl")

    # Save best hyper-parameters
    with open("../role_B/results/movielens-1m/svd_best_params.pkl", "wb") as f:
        pickle.dump(best, f)
    print("    Saved → role_B/results/movielens-1m/svd_best_params.pkl")