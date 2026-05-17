from Bio import SeqIO

fasta_path = r"C:\Users\LN\Desktop\ElicitorScreen\data\Harpin_Ea_len_filtered.fasta"
out_path = r"C:\Users\LN\Desktop\ElicitorScreen\data\proteinbert_seq_ids.txt"

seq_ids = [rec.id for rec in SeqIO.parse(fasta_path, "fasta")]
seq_ids = ["Harpin"] + seq_ids  

with open(out_path, "w", encoding="utf-8") as f:
    for sid in seq_ids:
        f.write(sid + "\n")

print("Saved IDs:", len(seq_ids))
print("Saved to:", out_path)