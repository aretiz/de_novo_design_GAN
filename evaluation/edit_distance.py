import matplotlib.pyplot as plt
import numpy as np
import Levenshtein
import random
from scipy.stats import gaussian_kde

translation_table = {
    'TTT': 'F', 'TTC': 'F',  # Phenylalanine (F)
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',  # Leucine (L)
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',  # Isoleucine (I)
    'ATG': 'M',  # Methionine (M) - Start codon
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',  # Valine (V)
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',  # Serine (S)
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',  # Proline (P)
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',  # Threonine (T)
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',  # Alanine (A)
    'TAT': 'Y', 'TAC': 'Y',  # Tyrosine (Y)
    'TAA': '', 'TAG': '', 'TGA': '',# '***': '#', # Stop codons or placeholders ('X')
    'CAT': 'H', 'CAC': 'H',  # Histidine (H)
    'CAA': 'Q', 'CAG': 'Q',  # Glutamine (Q)
    'AAT': 'N', 'AAC': 'N',  # Asparagine (N)
    'AAA': 'K', 'AAG': 'K',  # Lysine (K)
    'GAT': 'D', 'GAC': 'D',  # Aspartic Acid (D)
    'GAA': 'E', 'GAG': 'E',  # Glutamic Acid (E)
    'TGT': 'C', 'TGC': 'C',  # Cysteine (C)
    'TGG': 'W',  # Tryptophan (W)
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',  # Arginine (R)
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',  # Glycine (G)
}

def translate_dna_to_protein(dna_sequence, translation_table):
    protein_sequence = []
    codon = ""

    for nucleotide in dna_sequence:
        codon += nucleotide

        # Check if we have a complete codon
        if len(codon) == 3:
            # Translate the codon or use 'X' if not found in the table
            amino_acid = translation_table.get(codon, 'X')
            protein_sequence.append(amino_acid)
            codon = ""

    return ''.join(protein_sequence)

# Function to calculate normalized edit distance
def normalized_edit_distance(seq1, seq2):
    max_length = max(len(seq1), len(seq2))
    return Levenshtein.distance(seq1, seq2) / max_length

real_data = []

# Load real data
with open("./AMP_dataset.fa", "r") as file:   
    sequence = ""
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        # Split the line into DNA sequence and number
        parts = line.split('\t')
        sequence, number = parts
        if number == '1':
            real_data.append(translate_dna_to_protein(sequence, translation_table))

# print(len(real_data))
real_data = [seq[1:] for seq in real_data]

gen_sequences = []
with open("FBGAN_protein_sequences.fasta", 'r') as file:
    sequence = ''
    for line in file:
        line = line.strip()
        if line.startswith('>'):
            if sequence:
                gen_sequences.append(sequence)
                sequence = ''
        else:
            sequence += line
    if sequence:  # Append the last sequence
        gen_sequences.append(sequence)

print('Total selected sequences:', len(gen_sequences)) 

# Calculate within-group distances for real sequences
real_distances_within_group = []

selected_real_sequences = real_data

# Calculate distances among selected real sequences
for i in range(len(selected_real_sequences)):
    for j in range(i+1, len(selected_real_sequences)):
        distance_ij = normalized_edit_distance(selected_real_sequences[i], selected_real_sequences[j])
        real_distances_within_group.append(distance_ij)

# Calculate within-group distances for generated sequences
generated_distances_within_group = []

# Randomly select sequences
selected_generated_sequences = random.sample(gen_sequences, 2600)
# selected_generated_sequences = gen_sequences

# Calculate distances among selected generated sequences
for i in range(len(selected_generated_sequences)):
    for j in range(i+1, len(selected_generated_sequences)):
        distance_ij = normalized_edit_distance(selected_generated_sequences[i], selected_generated_sequences[j])
        generated_distances_within_group.append(distance_ij)

plt.figure(figsize=(8, 6))

plt.hist(real_distances_within_group, bins=50, alpha=0.5, color='blue', label='Real Data Within Group', density=True, range=(0.5, 1))
plt.hist(generated_distances_within_group, bins=50, alpha=0.5, color='orange', label='Generated Data Within Group', density=True, range=(0.5, 1))
plt.ylim(0, 18)

plt.xlabel('Normalized Edit Distance', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.title('FBGAN-ESM2', fontsize=16, fontweight='bold')
plt.legend(fontsize=16)  # Adjusting legend font size
plt.grid(True)

plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=13, fontweight='bold')

plt.show()
