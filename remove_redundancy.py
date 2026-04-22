import os
import random
from Bio import SeqIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


sample_size = 2000
similarity_threshold = 0.9
k = 3
random.seed(42)


print("Loading sequences...")
records = list(SeqIO.parse("splice_windows.fasta", "fasta"))

if len(records) == 0:
    raise ValueError("splice_windows.fasta is empty.")

sample_size = min(sample_size, len(records))
records = random.sample(records, sample_size)
print("Sampled sequences:", len(records))

def kmer_vector(sequence, k):
    vec = {}
    seq = str(sequence).upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        vec[kmer] = vec.get(kmer, 0) + 1
    return vec


vocab = set()
kmer_dicts = []

for record in records:
    vec = kmer_vector(record.seq, k)
    kmer_dicts.append(vec)
    vocab.update(vec.keys())

vocab = list(vocab)
vocab_index = {kmer:i for i,kmer in enumerate(vocab)}

matrix = np.zeros((len(records), len(vocab)))

for i, vec in enumerate(kmer_dicts):
    for kmer, count in vec.items():
        matrix[i][vocab_index[kmer]] = count


norms = np.linalg.norm(matrix, axis=1, keepdims=True)
matrix = matrix / np.maximum(norms, 1e-10)

print("Computing cosine similarity...")
similarity_matrix = cosine_similarity(matrix)

selected_indices = []

for i in range(len(records)):
    redundant = False
    for j in selected_indices:
        if similarity_matrix[i][j] > similarity_threshold:
            redundant = True
            break
    if not redundant:
        selected_indices.append(i)

with open("non_redundant_sequences.fasta", "w") as outfile:
    for idx in selected_indices:
        outfile.write(f">{records[idx].id}\n{records[idx].seq}\n")

print("Non-redundant sequences:", len(selected_indices))
print("Reduction percentage:",
      round((1 - len(selected_indices)/sample_size) * 100, 2), "%")