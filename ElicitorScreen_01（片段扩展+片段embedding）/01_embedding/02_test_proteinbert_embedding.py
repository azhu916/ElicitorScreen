import numpy as np
from Bio import SeqIO

fasta_path = r"C:\Users\LN\Desktop\ElicitorScreen\data\Harpin_Ea_len_filtered.fasta"

# 读取数据
records = []
for record in SeqIO.parse(fasta_path, "fasta"):
    records.append((record.id, str(record.seq)))

seq_ids = [r[0] for r in records]
sequences = [r[1] for r in records]

# 插入Harpin_Ea_D1_core蛋白
Harpin = """GGGNQNDTVNQLAGLLTGMMMMMS"""
sequences = [Harpin] + sequences
seq_ids = ["Harpin"] + seq_ids

# proteinbert导入
from proteinbert import load_pretrained_model
from proteinbert import InputEncoder

# 1) 加载预训练模型
model_generator, input_encoder = load_pretrained_model()

# 2) 设定最大长度
MAX_SEQ_LEN = 1024

# 3) 编码输入
X = input_encoder.encode_X(sequences, seq_len=MAX_SEQ_LEN)

# 4) 推理得到 embedding
Y = model_generator.create_model(seq_len=MAX_SEQ_LEN).predict(X, batch_size=8)

# 5) 保存
Y_seq = Y[0]    # shape: (856, 1024, 26) 序列级别embedding
Y_global = Y[1] # shape: (856, 128) 全局embedding

np.save("proteinbert_embeddings_seq.npy", Y_seq)
np.save("proteinbert_embeddings_global.npy", Y_global)

print("Seq embeddings shape:", Y_seq.shape)
print("Global embeddings shape:", Y_global.shape)
