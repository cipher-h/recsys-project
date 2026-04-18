import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# =========================
# Metric Functions 
# =========================
def hit_rate_at_k(predictions, ground_truth, K=10, users=None):
    hits, total = 0, 0
    for user_id, rec_items in predictions.items():
        if users is not None and user_id not in users: continue
        if user_id not in ground_truth or not ground_truth[user_id]: continue
        
        hits += int(len(set(rec_items[:K]) & ground_truth[user_id]) > 0)
        total += 1
    return hits / total if total > 0 else 0

def ndcg_at_k(predictions, ground_truth, K=10, users=None):
    scores = []
    for user_id, rec_items in predictions.items():
        if users is not None and user_id not in users: continue
        if user_id not in ground_truth or not ground_truth[user_id]: continue
        
        gt = ground_truth[user_id]
        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(rec_items[:K]) if item in gt)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt), K)))
        scores.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(scores) if scores else 0

def map_at_k(predictions, ground_truth, K=10, users=None):
    ap_list = []
    for user_id, rec_items in predictions.items():
        if users is not None and user_id not in users: continue
        if user_id not in ground_truth or not ground_truth[user_id]: continue
        
        gt = ground_truth[user_id]
        hits, ap = 0, 0
        for i, item in enumerate(rec_items[:K], 1):
            if item in gt:
                hits += 1
                ap += hits / i
        ap_list.append(ap / min(len(gt), K))
    return np.mean(ap_list) if ap_list else 0

# =========================
# Data Processing
# =========================
def get_ground_truth(test_df, label_col="label"):
    gt = {}
    pos = test_df[test_df[label_col] == 1]
    for user_id, group in pos.groupby("user_id"):
        gt[user_id] = set(group["item_id"].tolist())
    return gt

def get_cold_start_users(train_df, threshold=5):
    user_counts = train_df.groupby("user_id").size()
    cold_users = user_counts[user_counts <= threshold].index.tolist()
    return set(cold_users)

# =========================
# Evaluation Core (修复区)
# =========================
def calculate_metrics(preds_dict, gt, K, users=None):
    return {
        "HR": hit_rate_at_k(preds_dict, gt, K, users),
        "NDCG": ndcg_at_k(preds_dict, gt, K, users),
        "MAP": map_at_k(preds_dict, gt, K, users)
    }

def evaluate_model(pred_path, gt, name, Ks=[5,10,20], target_users=None):
    """
    修改点：严格根据传入的 target_users 计算指标，并直接返回该群体的结果。
    """
    with open(pred_path, "rb") as f:
        preds = pickle.load(f)

    preds_list = preds if isinstance(preds, list) else [preds]
    results = {}
    
    for K in Ks:
        # 如果 target_users 是空集，计算出的所有指标都会是 0
        seed_metrics = [calculate_metrics(p, gt, K, target_users) for p in preds_list]
        
        results[K] = {
            "HR": np.mean([m["HR"] for m in seed_metrics]),
            "NDCG": np.mean([m["NDCG"] for m in seed_metrics]),
            "MAP": np.mean([m["MAP"] for m in seed_metrics]),
            "HR_std": np.std([m["HR"] for m in seed_metrics]),
            "NDCG_std": np.std([m["NDCG"] for m in seed_metrics]),
            "MAP_std": np.std([m["MAP"] for m in seed_metrics]),
        }
    return results

def batch_evaluate(model_paths, gt, train_df=None, Ks=[5,10,20], plot=True, title="Model Comparison"):
    all_results = {}
    for name, path in model_paths.items():
        all_results[name] = evaluate_model(path, gt, name, Ks)
    return all_results

# =========================
# Visualization Functions for Plots
# =========================
def plot_overall_performance(results_dict, dataset_name, output_dir, Ks=[5, 10, 20]):
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ["HR", "NDCG", "MAP"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for i, (model_name, model_results) in enumerate(results_dict.items()):
            y_values = [model_results[K][metric] for K in Ks]
            ax.plot(Ks, y_values, marker=markers[i], color=colors[i], 
                    linewidth=2, markersize=8, label=model_name)
        
        ax.set_xticks(Ks)
        ax.set_xlabel("K", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric}@K on {dataset_name}", fontsize=14)
        if idx == 0:
            ax.legend(fontsize=10, loc='upper left')
            
    plt.tight_layout()
    safe_name = dataset_name.lower().replace(' ', '_').replace('.', '')
    output_path = os.path.join(output_dir, f"{safe_name}_overall.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_path}")

def plot_cold_start_bar(results_dict, dataset_name, output_dir, K=10):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    models = list(results_dict.keys())
    hr_values = [results_dict[m][K]["HR"] for m in models]
    ndcg_values = [results_dict[m][K]["NDCG"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, hr_values, width, label=f'HR@{K}', color='#1f77b4')
    rects2 = ax.bar(x + width/2, ndcg_values, width, label=f'NDCG@{K}', color='#ff7f0e')
    
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title(f'Cold-Start Users Performance on {dataset_name} (K={K})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    
    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    safe_name = dataset_name.lower().replace(' ', '_').replace('.', '')
    output_path = os.path.join(output_dir, f"{safe_name}_cold_start.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_path}")

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    output_dir = "recsys-project/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据集
    train_ml = pd.read_csv('recsys-project/data/splits/movielens-1m/train.csv')
    test_ml = pd.read_csv('recsys-project/data/splits/movielens-1m/test.csv')
    gt_ml = get_ground_truth(test_ml)
    
    train_lfm = pd.read_csv('recsys-project/data/splits/lastfm/train.csv')
    test_lfm = pd.read_csv('recsys-project/data/splits/lastfm/test.csv')
    gt_lfm = get_ground_truth(test_lfm)
    
    ml_models = {
        'Popularity': 'recsys-project/predictions/movielens-1m/popularity.pkl',
        'SVD': 'recsys-project/predictions/movielens-1m/svd_all_seeds.pkl',
        'NeuMF': 'recsys-project/predictions/movielens-1m/neumf_all_seeds.pkl',
        'TwoTower': 'recsys-project/predictions/movielens-1m/two_tower_all_seeds.pkl'
    }
    
    lfm_models = {
        'Popularity': 'recsys-project/predictions/lastfm/popularity.pkl',
        'ALS': 'recsys-project/predictions/lastfm/als_all_seeds.pkl',
        'NeuMF': 'recsys-project/predictions/lastfm/neumf_all_seeds.pkl',
        'TwoTower': 'recsys-project/predictions/lastfm/two_tower_all_seeds.pkl',
        'SASRec': 'recsys-project/predictions/lastfm/sasrec_all_seeds.pkl'
    }

    print("\n" + "="*60 + "\nCOMPUTING EVALUATIONS\n" + "="*60)
    
    # 1. 计算 MovieLens 1M (全局 + 冷启动)
    ml_overall, ml_cold = {}, {}
    cold_users_ml = get_cold_start_users(train_ml, threshold=5)
    
    print("\n[MovieLens 1M] - All Users")
    for name, path in ml_models.items():
        ml_overall[name] = evaluate_model(path, gt_ml, name, [5, 10, 20], target_users=None)
        # 【关键修复】：补回冷启动计算！
        ml_cold[name] = evaluate_model(path, gt_ml, name, [5, 10, 20], target_users=cold_users_ml)
        for K in [5, 10, 20]:
            r = ml_overall[name][K]
            print(f"  {name} @{K}: HR={r['HR']:.4f}±{r['HR_std']:.4f}, "
                  f"NDCG={r['NDCG']:.4f}±{r['NDCG_std']:.4f}, "
                  f"MAP={r['MAP']:.4f}±{r['MAP_std']:.4f}")
    
    # 2. 计算 Last.fm (全局 + 冷启动)
    lfm_overall, lfm_cold = {}, {}
    cold_users_lfm = get_cold_start_users(train_lfm, threshold=5)
    
    print("\n[Last.fm] - All Users")
    for name, path in lfm_models.items():
        lfm_overall[name] = evaluate_model(path, gt_lfm, name, [5, 10, 20], target_users=None)
        # 【关键修复】：补回冷启动计算！
        lfm_cold[name] = evaluate_model(path, gt_lfm, name, [5, 10, 20], target_users=cold_users_lfm)
        for K in [5, 10, 20]:
            r = lfm_overall[name][K]
            print(f"  {name} @{K}: HR={r['HR']:.4f}±{r['HR_std']:.4f}, "
                  f"NDCG={r['NDCG']:.4f}±{r['NDCG_std']:.4f}, "
                  f"MAP={r['MAP']:.4f}±{r['MAP_std']:.4f}")

    print("\n" + "="*60 + "\nGENERATING PLOTS\n" + "="*60)
    
    # 画全局图
    plot_overall_performance(ml_overall, "MovieLens 1M", output_dir, Ks=[5, 10, 20])
    plot_overall_performance(lfm_overall, "Last.fm", output_dir, Ks=[5, 10, 20])
    
    # 画冷启动图 (增加拦截逻辑：只有用户数 > 0 才画图)
    if len(cold_users_ml) > 0:
        plot_cold_start_bar(ml_cold, "MovieLens 1M", output_dir, K=10)
    else:
        print(f"⚠ 跳过绘制 MovieLens 1M 的冷启动图，因为冷启动用户数量为 {len(cold_users_ml)}。请手动删除旧图（如果有的话）。")
        
    if len(cold_users_lfm) > 0:
        plot_cold_start_bar(lfm_cold, "Last.fm", output_dir, K=10)
    else:
        print(f"⚠ 跳过绘制 Last.fm 的冷启动图。")
    
    print("\n" + "="*60 + "\nSAVING RESULTS TO CSV\n" + "="*60)
    
    # MovieLens 1M CSV (K=10 only)
    ml_rows = []
    for model_name in ml_models.keys():
        K = 10
        r = ml_overall[model_name][K]
        ml_rows.append({
            'Dataset': 'MovieLens 1M',
            'Model': model_name,
            'HR@10': f"{r['HR']:.4f}±{r['HR_std']:.4f}",
            'NDCG@10': f"{r['NDCG']:.4f}±{r['NDCG_std']:.4f}",
            'MAP@10': f"{r['MAP']:.4f}±{r['MAP_std']:.4f}"
        })
    
    ml_df = pd.DataFrame(ml_rows)
    ml_csv_path = os.path.join(output_dir, 'movielens-1m_results.csv')
    ml_df.to_csv(ml_csv_path, index=False)
    print(f"✓ Saved: {ml_csv_path}")
    
    # Last.fm CSV (K=10 only)
    lfm_rows = []
    for model_name in lfm_models.keys():
        K = 10
        r = lfm_overall[model_name][K]
        lfm_rows.append({
            'Dataset': 'Last.fm',
            'Model': model_name,
            'HR@10': f"{r['HR']:.4f}±{r['HR_std']:.4f}",
            'NDCG@10': f"{r['NDCG']:.4f}±{r['NDCG_std']:.4f}",
            'MAP@10': f"{r['MAP']:.4f}±{r['MAP_std']:.4f}"
        })
    
    lfm_df = pd.DataFrame(lfm_rows)
    lfm_csv_path = os.path.join(output_dir, 'lastfm_results.csv')
    lfm_df.to_csv(lfm_csv_path, index=False)
    print(f"✓ Saved: {lfm_csv_path}")
    
    # Combined CSV
    combined_df = pd.concat([ml_df, lfm_df], ignore_index=True)
    combined_csv_path = os.path.join(output_dir, 'all_results.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"✓ Saved: {combined_csv_path}")
        
    print("\n" + "="*60 + "\nEVALUATION COMPLETE\n" + "="*60)