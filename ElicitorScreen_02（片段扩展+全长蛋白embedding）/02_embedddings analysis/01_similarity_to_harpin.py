import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ====== 路径：按你自己的文件位置改 ======
EMB_PATH = r"C:\Users\LN\Desktop\ElicitorScreen\ElicitorScreen_02（片段扩展+全长蛋白embedding）\data\proteinbert_embeddings_global.npy"
ID_PATH  = r"C:\Users\LN\Desktop\ElicitorScreen\ElicitorScreen_02（片段扩展+全长蛋白embedding）\data\proteinbert_seq_ids.txt"  
OUT_CSV  = r"C:\Users\LN\Desktop\ElicitorScreen\ElicitorScreen_02（片段扩展+全长蛋白embedding）\data\results_similarity_to_harpin.csv"

TOPK = 20
HARPIN_INDEX = 0  # 你约定 Harpin 在第0行

# ====== 1) 读取 embedding ======
embeddings = np.load(EMB_PATH)
print("Embeddings shape:", embeddings.shape)

# ====== 2) 读取 seq_ids ======
with open(ID_PATH, "r", encoding="utf-8") as f:
    seq_ids = [line.strip() for line in f]

if embeddings.shape[0] != len(seq_ids):
    raise ValueError(f"Row mismatch: embeddings rows={embeddings.shape[0]} but seq_ids={len(seq_ids)}")

print("Total seqs:", len(seq_ids))
print("Harpin ID:", seq_ids[HARPIN_INDEX])

# ====== 3) 计算与 Harpin 的 cosine similarity ======
harpin_vec = embeddings[HARPIN_INDEX].reshape(1, -1)          # (1, D)
sim = cosine_similarity(harpin_vec, embeddings)[0]            # (N,)

# 去掉自己
sim[HARPIN_INDEX] = -1.0

# ====== 4) 排名 ======
rank_idx = np.argsort(sim)[::-1]

# ====== 5) 输出 TopK ======
print(f"\nTop {TOPK} similar proteins to Harpin:")
for i in rank_idx[:TOPK]:
    print(i, seq_ids[i], float(sim[i]))

# ====== 6) 保存结果表 ======
df = pd.DataFrame({
    "index": np.arange(len(seq_ids)),
    "seq_id": seq_ids,
    "cosine_sim_to_harpin": sim
}).sort_values("cosine_sim_to_harpin", ascending=False)

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("\nSaved:", OUT_CSV)

# ====== 7) 绘图 ======
import matplotlib.pyplot as plt

print("Max similarity:", np.max(sim))
print("Mean similarity:", np.mean(sim[1:]))
print("Median similarity:", np.median(sim[1:]))

plt.hist(sim[1:], bins=50)
plt.title("Similarity to Harpin")
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.show()
