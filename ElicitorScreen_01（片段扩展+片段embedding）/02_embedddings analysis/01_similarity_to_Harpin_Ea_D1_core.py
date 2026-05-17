import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1️⃣ 读取 embedding
embeddings = np.load(r"C:\Users\LN\Desktop\ElicitorScreen\data\proteinbert_embeddings_global.npy")

# 2️⃣ 读取序列 ID
with open(r"C:\Users\LN\Desktop\ElicitorScreen\data\proteinbert_seq_ids.txt", "r") as f:
    seq_ids = [line.strip() for line in f]

print("Embedding shape:", embeddings.shape)
print("Total sequences:", len(seq_ids))

# 3️⃣ Harpin_Ea_D1_core 是第一个
harpin_vec = embeddings[0].reshape(1, -1)

# 4️⃣ 计算 cosine similarity
sim = cosine_similarity(harpin_vec, embeddings)[0]

# 5️⃣ 去掉自己
sim[0] = -1

# 6️⃣ 排序
rank_idx = np.argsort(sim)[::-1]

# 7️⃣ 打印 Top10 相似蛋白
print("\nTop 10 similar to Harpin_Ea_D1_core:\n")

for i in rank_idx[:10]:
    print(f"{seq_ids[i]}\t{sim[i]:.6f}")

# 8️⃣ 绘图
import matplotlib.pyplot as plt

print("Max similarity:", np.max(sim))
print("Mean similarity:", np.mean(sim[1:]))
print("Median similarity:", np.median(sim[1:]))

plt.hist(sim[1:], bins=50)
plt.title("Similarity to Harpin_Ea_D1_core")
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.show()