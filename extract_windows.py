import os
import gzip
from Bio import SeqIO

# ─── Always run from script folder ───────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

print("Working directory:", os.getcwd())

# ─── Auto-detect genome file (.fa or .fa.gz) ─────────────────
genome_file = None
for f in os.listdir():
    if f.startswith("Homo_sapiens.GRCh38") and (f.endswith(".fa") or f.endswith(".fa.gz")):
        genome_file = f
        break

if genome_file is None:
    raise FileNotFoundError("Genome FASTA (.fa or .fa.gz) not found in this folder.")

print("Using genome file:", genome_file)

# ─── Load genome ──────────────────────────────────────────────
if genome_file.endswith(".gz"):
    handle = gzip.open(genome_file, "rt")
else:
    handle = open(genome_file, "r")

print("Loading genome into memory...")
genome = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
handle.close()
print("Genome loaded.")

# ─── Input / Output ───────────────────────────────────────────
input_file = "clinvar.vcf"
output_file = "splice_windows.fasta"
window_size = 100

if not os.path.exists(input_file):
    raise FileNotFoundError(f"{input_file} not found in this folder.")

count = 0

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if line.startswith("#"):
            continue

        cols = line.strip().split("\t")
        chrom = cols[0]
        pos = int(cols[1])

        if chrom in genome:
            seq = genome[chrom].seq
            start = pos - window_size
            end = pos + window_size

            if start > 0 and end < len(seq):
                window_seq = seq[start:end]
                outfile.write(f">{chrom}_{pos}\n{window_seq}\n")
                count += 1

print("Total sequence windows extracted:", count)
print("Output file created:", output_file)