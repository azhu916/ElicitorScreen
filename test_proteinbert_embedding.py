import numpy as np
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import (
    get_model_with_hidden_layers_as_outputs
)

# =========================
# 0. 固定参数
# =========================
SEQ_LEN = 128   # 故意放大，容纳不同长度

# =========================
# 1. 两条不同长度的序列
# =========================
sequences = [
    # 短序列（~60 aa）
    "MSLNTSGLGASTMQISIGGAGGNNGLLGTSRQNAGLGGNSALGLGGGNQNDTVNQLAGLL",

    # 长序列（~150 aa）
    "MSLNTSGLGASTMQISIGGAGGNNGLLGTSRQNAGLGGNSALGLGGGNQNDTVNQLAGLL"
    "TGMMMMMSMMGGGGLMGGGLGGGLGNGLGGSGGLGEGLSNALNDMLGGSLNTLGSKGGNN"
    "TTSTTNSPLDQALGINSTSQNDDSTSGTDSTSDSSDPMQQLLKMFSEIMQSLFGDGQDGT"
]

print("Sequence lengths:", [len(s) for s in sequences])
print("Using SEQ_LEN:", SEQ_LEN)

# =========================
# 2. 加载模型
# =========================
print("Loading ProteinBERT...")
model_generator, tokenizer = load_pretrained_model()

# =========================
# 3. 编码
# =========================
encoded = tokenizer.encode_X(sequences, seq_len=SEQ_LEN)

# 兜底处理（你这个版本需要）
if isinstance(encoded, (list, tuple)):
    model_input = encoded
    token_tensor = encoded[0]
else:
    model_input = encoded
    token_tensor = encoded

print("Token tensor shape:", token_tensor.shape)  # (batch, SEQ_LEN)

# =========================
# 4. 构建模型
# =========================
model = model_generator.create_model(seq_len=SEQ_LEN)
model = get_model_with_hidden_layers_as_outputs(model)
print("Model built.")

# =========================
# 5. 前向计算
# =========================
outputs = model.predict(model_input)

# 找 3D hidden states
last_hidden = None
for out in outputs:
    if hasattr(out, "ndim") and out.ndim == 3:
        last_hidden = out
        break

if last_hidden is None:
    raise RuntimeError("No 3D hidden state found")

print("Hidden state shape:", last_hidden.shape)
# (batch, SEQ_LEN, hidden_dim)

# =========================
# 6. Pooling（关键）
# =========================
embeddings = last_hidden.mean(axis=1)   # (batch, hidden_dim)

print("Final embedding matrix shape:", embeddings.shape)

for i, emb in enumerate(embeddings):
    print(f"Sequence {i} embedding shape:", emb.shape)

# =========================
# 7. 保存
# =========================
np.save("test_embeddings_multi.npy", embeddings)
print("Saved to test_embeddings_multi.npy")