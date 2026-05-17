from Bio import SeqIO

# 输入输出路径
input_fasta = r"C:\Users\LN\Desktop\毕业设计——植物疫苗与受体的相互关系及免疫活性研究\标准配对\Harpin Ea——HIPM\Harpin Ea\harpin_Ea_D1 core 同源扩展\去冗余\Harpin_Ea_cdhit90.fasta"
output_fasta = r"C:\Users\LN\Desktop\ElicitorScreen\Harpin_Ea_len_filtered.fasta"

# 长度阈值
MIN_LEN = 30
MAX_LEN = 1022

kept = []
removed = []

for record in SeqIO.parse(input_fasta, "fasta"):
    seq = str(record.seq)
    if MIN_LEN <= len(seq) <= MAX_LEN:
        kept.append(record)
    else:
        removed.append(record)

SeqIO.write(kept, output_fasta, "fasta")

print("Total sequences:", len(kept) + len(removed))
print("Kept:", len(kept))
print("Removed:", len(removed))
print("Saved to:", output_fasta)