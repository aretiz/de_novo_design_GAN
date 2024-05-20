from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np

def calculate_aa_composition(sequences):
    aa_counts = {}
    total_aa = 0
    for sequence in sequences:
        total_aa += len(sequence)
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
    aa_composition = {aa: count / total_aa for aa, count in sorted(aa_counts.items())}
    return aa_composition

# Function to plot amino acid composition
def plot_aa_composition(ax, aa_composition, title, ylabel=True):
    aa = list(aa_composition.keys())
    frequencies = list(aa_composition.values())

    # Define colors for specific amino acids
    colors = ['red' if aa in ['C', 'D', 'E', 'K', 'R'] else 'blue' for aa in aa]

    ax.bar(aa, frequencies, color=colors)
    ax.set_title(title, weight='bold', size=16)
    ax.set_ylabel('Frequency' if ylabel else '', weight='bold', size=16)
    ax.set_xticks(aa)  # Set x-ticks for all amino acids

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        
    ax.set_ylim(0, 0.25)  # Set y-axis limit

    # Print frequencies with model name
    print(f"Frequencies for {title}:")
    for aa, freq in zip(aa, frequencies):
        print(f"Model: {title}, Amino Acid: {aa}, Frequency: {freq:.4f}")
    print()
    
# Read real sequences from FASTA file
real_sequences = {}
files = ['real_above_0.8.fasta', 'Gupta_above_0.8.fasta', 'kmers_above_0.8.fasta','MLP_above_0.8.fasta', 'AMPGAN_above_0.8.fasta', 'hydrAMP_above_0.8.fasta']
names = ['Real data', 'FBGAN', 'FBGAN-kmers']

# Create subplots for the first three plots
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

# Calculate amino acid composition for the first three datasets and plot
for i, file in enumerate(files[:3]):
    sequences = []
    with open(file) as file_handle:
        for record in SeqIO.parse(file_handle, 'fasta'):
            sequence = str(record.seq)
            sequences.append(sequence)
    aa_composition = calculate_aa_composition(sequences)
    plot_aa_composition(axes1[i], aa_composition, f'{names[i]}', ylabel=(i==0))


# Adjust layout and spacing for the first figure
plt.tight_layout()
plt.show()

# Create subplots for the last three plots
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
names = ['FBGAN-ESM2', 'AMPGAN', 'hydrAMP']

# Calculate amino acid composition for the last three datasets and plot
for i, file in enumerate(files[3:]):
    sequences = []
    with open(file) as file_handle:
        for record in SeqIO.parse(file_handle, 'fasta'):
            sequence = str(record.seq)
            sequences.append(sequence)
    print(f"Loaded {len(sequences)} sequences from {file}")  # Print number of sequences loaded
    aa_composition = calculate_aa_composition(sequences)
    plot_aa_composition(axes2[i], aa_composition, f'{names[i]}', ylabel=(i==0))

# Adjust layout and spacing for the second figure
plt.tight_layout()
plt.show()

# Function to calculate Kullback-Leibler Divergence (KLD)
def kld(p, q):
    return np.sum(p * np.log(p / q))

# List of file names
files = ['real_above_0.8.fasta', 'Gupta_above_0.8.fasta', 'kmers_above_0.8.fasta', 'MLP_above_0.8.fasta', 'AMPGAN_above_0.8.fasta', 'hydrAMP_above_0.8.fasta']

# Read sequences and calculate amino acid composition
real_aa_composition = calculate_aa_composition(SeqIO.parse('real_above_0.8.fasta', 'fasta'))

for file in files[1:]:
    generated_aa_composition = calculate_aa_composition(SeqIO.parse(file, 'fasta'))
    kld_score = kld(np.array(list(real_aa_composition.values())), np.array(list(generated_aa_composition.values())))
    print(f"KLD between real and {file}: {kld_score}")