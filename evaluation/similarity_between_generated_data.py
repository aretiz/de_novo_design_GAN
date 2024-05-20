from Bio import pairwise2
from Bio.Seq import Seq
import random

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
    'TAA': '', 'TAG': '', 'TGA': '', # Stop codons 
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

# Function to calculate overall sequence similarity as a percentage
def overall_sequence_similarity_percentage(sequences):
    total_similarity = 0
    total_combinations = 0

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1 = Seq(sequences[i])
            seq2 = Seq(sequences[j])
            similarity = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0][2]
            max_length = max(len(seq1), len(seq2))
            total_similarity += (similarity / max_length)
            total_combinations += 1

    overall_similarity_percentage = (total_similarity / total_combinations) * 100
    return overall_similarity_percentage

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

# selected_generated_sequences = random.sample(gen_sequences, 4947)
selected_generated_sequences = gen_sequences

overall_similarity = overall_sequence_similarity_percentage(selected_generated_sequences)
print(f"Overall sequence similarity: {overall_similarity:.2f}%")