import re
import random

translation_table1 = {
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

translation_table2 = {
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
    'TAA': '_', 'TAG': '_', 'TGA': '_',# '***': '#', # Stop codons or placeholders ('X')
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

def write_sequences_to_fasta(sequences, output_file):
    with open(output_file, "w") as fasta_file:
        for i, sequence in enumerate(sequences):
            header = f">Sequence_{i + 1}\n"  # Customize the header as needed
            fasta_file.write(header)
            fasta_file.write(sequence + "\n")

selected_sequences = []

filename = f"./generated_samples_FBGAN_dataset.txt"

with open(filename, "r") as file:
    sequence = ""
    for line in file:
        parts = line.split('\n')
        seq = parts[0]
        # modified_seq = re.sub(r'P*$', '', seq)
        modified_seq = re.sub('P', '', seq)
        condition_met = (
                        len(modified_seq) >= 15
                        and len(modified_seq) % 3 == 0
                        and translate_dna_to_protein(modified_seq, translation_table2).startswith('M')
                        and translate_dna_to_protein(modified_seq, translation_table2).endswith('_')
                    )
        if condition_met:
            selected_sequences.append(translate_dna_to_protein(modified_seq, translation_table1))

print('Total selected sequences:', len(selected_sequences))
print("*"*100)

selected_sequences = [seq[1:] if seq.startswith("M") else seq for seq in selected_sequences]

random.seed(42)
selected_sequences = random.sample(selected_sequences, 4947)

write_sequences_to_fasta(selected_sequences, "FBGAN_protein_sequences.fasta")

unique_sequences = set()

# Create a list to store duplicates
duplicate_sequences = []

# Iterate through the protein sequences
for sequence in selected_sequences:
    if sequence not in unique_sequences:
        # If the sequence is not in the set, add it to the set (unique) and continue
        unique_sequences.add(sequence)
        # print(unique_sequences)
    else:
        # If the sequence is already in the set, add it to the duplicates list
        duplicate_sequences.append(sequence)

# Calculate the number of unique and duplicate sequences
num_unique_sequences = len(unique_sequences)
num_duplicate_sequences = len(duplicate_sequences)

# Print the results
print(f"Number of unique sequences: {num_unique_sequences}")
print(f"Number of duplicate sequences: {num_duplicate_sequences}")
print("*"*100)

train_sequences = []
with open("./data/dna_uniprot_under_50_reviewed.fasta", "r") as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        parts = line.split(' ')
        
        sequence = parts[0]    
        train_sequences.append(translate_dna_to_protein(sequence, translation_table1))
      
print('Number of traing sequences:', len(train_sequences))

# Remove duplicate 'M' characters at the beginning using a list comprehension
train_sequences = [seq[1:] if seq.startswith("M") else seq for seq in train_sequences]
num_common_strings = [item for item in train_sequences if item in selected_sequences]

if num_common_strings:
    print(f"Common strings found between train and generated sequences: {num_common_strings}")
else:
    print(f"Common strings found between train and generated sequences: 0")
