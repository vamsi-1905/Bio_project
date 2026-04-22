"""
pipeline.py — Step 1
Extracts 50/100/200bp windows for REF and ALT sequences.
Saves to parquet (fast, compressed, no SQL needed).

Requirements:
    pip install biopython pandas tqdm pyarrow

Usage:
    python pipeline.py
    (edit the paths at the top if needed)
"""



import os
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

# Set working directory
BASE_DIR = r"C:\Users\vamsi\Downloads\SEM 3\BIO\ibs_project_full\ibs_lab"
os.chdir(BASE_DIR)

# --- CONFIG ---
CSV_PATH  = os.path.join(BASE_DIR, "splice_dataset_full.csv")
FASTA_PATH = os.path.join(BASE_DIR, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
OUT_PATH  = os.path.join(BASE_DIR, "splice_windows.parquet")

WINDOWS = [50, 100, 200]  # multi-scale


def load_genome(fasta_path):
    print("Loading genome (this takes ~2-3 mins)...")
    genome = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    print(f"Genome loaded. Chromosomes found: {len(genome)}")
    return genome


def get_chrom_key(genome, chrom):
    """Try different chromosome name formats."""
    for candidate in [str(chrom), f"chr{chrom}", str(chrom).replace("chr", "")]:
        if candidate in genome:
            return candidate
    return None


def apply_mutation(ref_window, center, ref_allele, alt_allele):
    """Replace ref allele with alt allele at center position in window."""
    before  = ref_window[:center]
    after   = ref_window[center + len(ref_allele):]
    return before + str(alt_allele) + after


def extract_windows_for_variant(genome, chrom, pos, ref, alt):
    """
    Returns dict of window_size → (ref_seq, alt_seq).
    pos is 1-based (VCF/ClinVar convention).
    """
    key = get_chrom_key(genome, chrom)
    if key is None:
        return None

    genome_seq = genome[key].seq
    genome_len = len(genome_seq)
    pos_0      = int(pos) - 1    # convert to 0-based

    result = {}
    for w in WINDOWS:
        start  = max(0, pos_0 - w)
        end    = min(genome_len, pos_0 + w)
        center = pos_0 - start

        ref_window = str(genome_seq[start:end]).upper()

        # Sanity check
        extracted_ref = ref_window[center:center + len(str(ref))]
        if extracted_ref != str(ref).upper():
            # Mismatch — skip this window size
            continue

        alt_window = apply_mutation(ref_window, center, str(ref), str(alt))

        result[w] = (ref_window, alt_window)

    return result if result else None


def run():
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} variants from {CSV_PATH}")

    # Load genome
    genome = load_genome(FASTA_PATH)

    rows = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting windows"):
        windows = extract_windows_for_variant(
            genome,
            chrom = row["chrom"],
            pos   = row["position"],
            ref   = row["ref"],
            alt   = row["alt"]
        )

        if windows is None:
            skipped += 1
            continue

        entry = {
            "chrom"    : row["chrom"],
            "position" : row["position"],
            "ref"      : row["ref"],
            "alt"      : row["alt"],
            "label"    : row["label"],
        }

        for w in WINDOWS:
            if w in windows:
                entry[f"ref_seq_{w}"]  = windows[w][0]
                entry[f"alt_seq_{w}"]  = windows[w][1]
            else:
                entry[f"ref_seq_{w}"]  = None
                entry[f"alt_seq_{w}"]  = None

        rows.append(entry)

    result_df = pd.DataFrame(rows)

    # Train/val/test split — stratified
    from sklearn.model_selection import train_test_split

    trainval, test = train_test_split(
        result_df, test_size=0.2, random_state=42, stratify=result_df["label"]
    )
    train, val = train_test_split(
        trainval, test_size=0.25, random_state=42, stratify=trainval["label"]
    )

    result_df["split"] = "train"
    result_df.loc[val.index,  "split"] = "val"
    result_df.loc[test.index, "split"] = "test"

    # Save
    result_df.to_parquet(OUT_PATH, index=False)

    print(f"\nDone.")
    print(f"  Total processed : {len(result_df):,}")
    print(f"  Skipped         : {skipped:,}  (chrom not found or REF mismatch)")
    print(f"  Train           : {(result_df['split']=='train').sum():,}")
    print(f"  Val             : {(result_df['split']=='val').sum():,}")
    print(f"  Test            : {(result_df['split']=='test').sum():,}")
    print(f"  Saved to        : {OUT_PATH}")
    print(f"\nColumns: {result_df.columns.tolist()}")


if __name__ == "__main__":
    run()