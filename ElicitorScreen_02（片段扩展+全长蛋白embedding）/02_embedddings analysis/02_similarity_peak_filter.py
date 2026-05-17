"""
从 Harpin 相似度右侧峰中挑选10个代表性序列
策略：KMeans 聚类（k=10），取各簇中心最近点 → 多样性最佳
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ====== 路径配置（按需修改） ======
EMB_PATH = r"C:\Users\LN\Desktop\ElicitorScreen\ElicitorScreen_02（片段扩展+全长蛋白embedding）\data\proteinbert_embeddings_global.npy"
ID_PATH  = r"C:\Users\LN\Desktop\ElicitorScreen\ElicitorScreen_02（片段扩展+全长蛋白embedding）\data\proteinbert_seq_ids.txt"
SIM_CSV  = r"C:\Users\LN\Desktop\ElicitorScreen\ElicitorScreen_02（片段扩展+全长蛋白embedding）\data\results_similarity_to_harpin.csv"
OUT_CSV  = r"C:\Users\LN\Desktop\ElicitorScreen\ElicitorScreen_02（片段扩展+全长蛋白embedding）\data\representative_10_from_peak.csv"

HARPIN_INDEX = 0

# ====== 右侧峰范围（根据图形目视判断） ======
# 可选方案A：保守范围
PEAK_LOW  = 0.60
PEAK_HIGH = 0.75

# 可选方案B：宽松范围（取消注释以使用）
# PEAK_LOW  = 0.55
# PEAK_HIGH = 0.80

# ====== 1) 加载数据 ======
embeddings = np.load(EMB_PATH)
with open(ID_PATH, "r", encoding="utf-8") as f:
    seq_ids = [line.strip() for line in f]

sim_df = pd.read_csv(SIM_CSV, encoding="utf-8-sig")
sim_arr = sim_df.sort_values("index")["cosine_sim_to_harpin"].values

print(f"Total sequences: {len(seq_ids)}")
print(f"Peak range: [{PEAK_LOW}, {PEAK_HIGH}]")

# ====== 2) 筛选右侧峰内的序列 ======
peak_mask = (sim_arr >= PEAK_LOW) & (sim_arr <= PEAK_HIGH)
peak_mask[HARPIN_INDEX] = False   # 排除 Harpin 本身

peak_indices  = np.where(peak_mask)[0]
peak_embs     = embeddings[peak_indices]    # shape: (M, D)
peak_sim      = sim_arr[peak_indices]
peak_ids      = [seq_ids[i] for i in peak_indices]

print(f"Sequences in peak: {len(peak_indices)}")

# ====== 3) KMeans 聚类 → 取各簇代表 ======
K = 10
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(peak_embs)

representatives = []
for k in range(K):
    cluster_mask = labels == k
    cluster_embs = peak_embs[cluster_mask]
    cluster_orig_idx = peak_indices[cluster_mask]
    cluster_sim  = peak_sim[cluster_mask]
    cluster_ids  = [peak_ids[i] for i in np.where(cluster_mask)[0]]

    # 找离聚类中心最近的点（用欧氏距离）
    center = kmeans.cluster_centers_[k]
    dists  = np.linalg.norm(cluster_embs - center, axis=1)
    best   = np.argmin(dists)

    representatives.append({
        "cluster":           k,
        "seq_id":            cluster_ids[best],
        "original_index":    int(cluster_orig_idx[best]),
        "cosine_sim_to_harpin": float(cluster_sim[best]),
        "cluster_size":      int(cluster_mask.sum()),
        "dist_to_center":    float(dists[best]),
    })

rep_df = pd.DataFrame(representatives).sort_values("cosine_sim_to_harpin", ascending=False)

# ====== 4) 输出结果 ======
print("\n===== 10 Representative Sequences from Right Peak =====")
print(rep_df[["cluster", "seq_id", "cosine_sim_to_harpin", "cluster_size"]].to_string(index=False))

rep_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUT_CSV}")

# ====== 5) 可视化：在直方图上标记代表序列 ======
fig, ax = plt.subplots(figsize=(8, 5))

# 全局分布（灰色背景）
all_sim = sim_arr.copy()
all_sim[HARPIN_INDEX] = np.nan
ax.hist(all_sim, bins=50, color="steelblue", alpha=0.4, label="All sequences")

# 右侧峰（深色）
ax.hist(peak_sim, bins=30, color="steelblue", alpha=0.8, label=f"Peak [{PEAK_LOW}–{PEAK_HIGH}]")

# 代表序列（红色竖线）
for _, row in rep_df.iterrows():
    ax.axvline(row["cosine_sim_to_harpin"], color="red", linewidth=1.2, alpha=0.8)

ax.axvline(rep_df["cosine_sim_to_harpin"].iloc[0], color="red",
           linewidth=1.2, alpha=0.8, label="Representatives (n=10)")

ax.set_xlabel("Cosine similarity to Harpin")
ax.set_ylabel("Count")
ax.set_title("Representative Sequences from Right Peak (KMeans, k=10)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_CSV.replace(".csv", "_plot.png"), dpi=150)
plt.show()
print("Plot saved.")