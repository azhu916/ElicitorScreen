import numpy as np
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import (
    get_model_with_hidden_layers_as_outputs
)

# =========================
# 1. 两条不同长度序列
# =========================
sequences = [
    # 60 aa
    "MSLNTSGLGASTMQISIGGAGGNNGLLGTSRQNAGLGGNSALGLGGGNQNDTVNQLAGLL",

    # 180 aa
    "MSLNTSGLGASTMQISIGGAGGNNGLLGTSRQNAGLGGNSALGLGGGNQNDTVNQLAGLL"
    "TGMMMMMSMMGGGGLMGGGLGGGLGNGLGGSGGLGEGLSNALNDMLGGSLNTLGSKGGNN"
    "TTSTTNSPLDQALGINSTSQNDDSTSGTDSTSDSSDPMQQLLKMFSEIMQSLFGDGQDGT"
]

lengths = [len(s) for s in sequences]
SEQ_LEN = max(lengths) + 4   # 安全 buffer

print("Sequence lengths:", lengths)
print("Using SEQ_LEN:", SEQ_LEN)

# =========================
# 2. 加载模型
# =========================
model_generator, tokenizer = load_pretrained_model()

# =========================
# 3. 编码（现在一定不会炸）
# =========================
encoded = tokenizer.encode_X(sequences, seq_len=SEQ_LEN)

if isinstance(encoded, (list, tuple)):
    model_input = encoded
    token_tensor = encoded[0]
else:
    model_input = encoded
    token_tensor = encoded

print("Token tensor shape:", token_tensor.shape)  # (2, SEQ_LEN)

# =========================
# 4. 构建模型
# =========================
model = model_generator.create_model(seq_len=SEQ_LEN)
model = get_model_with_hidden_layers_as_outputs(model)

# =========================
# 5. 前向
# =========================
outputs = model.predict(model_input)

# 找 3D hidden states
last_hidden = None
for out in outputs:
    if hasattr(out, "ndim") and out.ndim == 3:
        last_hidden = out
        break

print("Hidden state shape:", last_hidden.shape)

# =========================
# 6. Pooling
# =========================
embeddings = last_hidden.mean(axis=1)

print("Final embedding matrix shape:", embeddings.shape)
for i, emb in enumerate(embeddings):
    print(f"Sequence {i} embedding shape:", emb.shape)
