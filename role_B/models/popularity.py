import pandas as pd
import pickle
from collections import Counter

class PopularityBaseline:
    def fit(self, train_df):
        # Only count the popularity of positive samples (label = 1)
        pos = train_df[train_df["label"] == 1]
        item_counts = Counter(pos["item_id"].tolist())
        self.popular_items = [item for item, _ in item_counts.most_common()]

        # Record the items each user has viewed (to exclude them during recommendation)
        self.user_seen = (
            train_df.groupby("user_id")["item_id"]
            .apply(set).to_dict()
        )

    def recommend(self, user_id, K=10):
        seen = self.user_seen.get(user_id, set())
        recs = [i for i in self.popular_items if i not in seen]
        return recs[:K]

    def recommend_all(self, test_users, K=10):
        return {uid: self.recommend(uid, K) for uid in test_users}


if __name__ == "__main__":
    for dataset in ["movielens-1m", "lastfm"]:
        print(f"\n>>> Running Popularity on {dataset}")
        train = pd.read_csv(f"../data/splits/{dataset}/train.csv")
        test  = pd.read_csv(f"../data/splits/{dataset}/test.csv")

        model = PopularityBaseline()
        model.fit(train)

        test_users = test["user_id"].unique().tolist()
        predictions = model.recommend_all(test_users, K=10)

        out_path = f"../predictions/{dataset}/popularity.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(predictions, f)
        print(f"    Saved {len(predictions)} users → {out_path}")
