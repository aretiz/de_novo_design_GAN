import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import random, re
from Bio import SeqIO
from modlamp.descriptors import GlobalDescriptor
from Bio.SeqUtils.ProtParam import ProteinAnalysis

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

def calculate_protein_descriptors(sequence):
    # Create a ProteinAnalysis object for the given sequence
    analysed_seq = ProteinAnalysis(sequence)
    
    # Calculate molecular weight
    mw = analysed_seq.molecular_weight()
    
    # Calculate aromaticity
    aromaticity = analysed_seq.aromaticity()
    
    # Calculate instability index
    instability_index = analysed_seq.instability_index()
    
    # Calculate isoelectric point
    pI = analysed_seq.isoelectric_point()
    
    flexibility = analysed_seq.flexibility()
    hydrophobicity = analysed_seq.gravy(scale='Fauchere') #BlackMould Fauchere

    return mw, aromaticity, instability_index, pI, flexibility, hydrophobicity

def calculate_protein_descriptors_and_features(filename):

    globdesc = GlobalDescriptor(filename)

    # --------------- Global Descriptor Calculations ---------------
    globdesc.length()  # sequence length
    globdesc.boman_index(append=True)  # Boman index
    globdesc.aromaticity(append=True)  # global aromaticity
    globdesc.aliphatic_index(append=True)  # aliphatic index
    globdesc.instability_index(append=True)  # instability index
    globdesc.calculate_charge(ph=7.0, amide=False, append=True)  # net charge
    globdesc.calculate_MW(amide=False, append=True)  # molecular weight

    # Save descriptor data to .csv file
    col_names2 = 'ID,Sequence,Length,BomanIndex,Aromaticity,AliphaticIndex,InstabilityIndex,Charge,MW'
    globdesc.save_descriptor('descriptors2.csv', header=col_names2)

    # Read descriptor data into a DataFrame
    df = pd.read_csv('descriptors2.csv')

    # Extract required columns
    column1 = df['Length'].tolist()
    column2 = df['MW'].tolist()
    column3 = df['Charge'].tolist()
    column4 = df['InstabilityIndex'].tolist()
    column5 = df['Aromaticity'].tolist()
    column6 = df['AliphaticIndex'].tolist()
    column7 = df['BomanIndex'].tolist()

    # Clean and convert data
    metric1 = [float(value.strip("b'")) for value in column1]                    	
    metric2 = [float(value.strip("b'")) for value in column2]                    	
    metric3 = [float(value.strip("b'")) for value in column3]                    	
    metric4 = [float(value.strip("b'")) for value in column4]                    	
    metric5 = [float(value.strip("b'")) for value in column5]                    	
    metric6 = [float(value.strip("b'")) for value in column6]                    	
    metric7 = [float(value.strip("b'")) for value in column7]       

    # Example data for the physicochemical features
    data = {
        'Length': metric1,
        # 'MW': metric2,
        'Charge': metric3,
        'InstabilityIndex': metric4,
        'Aromaticity': metric5,
        'AliphaticIndex': metric6,
        'BomanIndex': metric7,
    }

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    sequences = []
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences.append(str(record.seq))

    hydrophobicity_values = []
    pi_values = []
    for sequence in sequences:
        mw, aromaticity, instability_index, pI, flexibility, hydrophobicity = calculate_protein_descriptors(sequence)
        hydrophobicity_values.append(hydrophobicity)
        pi_values.append(pI)

    df['pI'] = pi_values
    df['HydrophobicityRatio'] = hydrophobicity_values

    return df

def plot_average_comparison(features_datasets, feature_names, labels, colors=None):
    """
    Plots the average of each metric for multiple datasets with standard error bars.

    Parameters:
        features_datasets (list of lists): A list containing lists of datasets to be plotted for each feature.
                                           Each inner list should contain the datasets to be plotted for a specific feature.
        feature_names (list of str): A list of feature names corresponding to the features_datasets.
        labels (list of str): Labels for the datasets.
        colors (list of str, optional): A list of colors to use for each dataset. If not provided, default colors will be used.
    """
    num_features = len(feature_names)
    num_datasets = len(features_datasets[0])

    # Calculate averages and standard errors for each dataset and feature
    avg_values = np.zeros((num_datasets, num_features))
    sem_values = np.zeros((num_datasets, num_features))
    for i in range(num_datasets):
        for j in range(num_features):
            avg_values[i, j] = np.mean(features_datasets[j][i])
            sem_values[i, j] = np.std(features_datasets[j][i]) / np.sqrt(len(features_datasets[j][i]))

    # Calculate real data averages
    real_avg_values = avg_values[0, :]
    plot_titles = ['Charge', 'pI', 'Aromaticity', 'Hydrophobicity Ratio']

    # Plotting
    fig, axes = plt.subplots(2,2, figsize=(14, 10))
    axes = axes.flatten()
    for i in range(num_features):
        ax = axes[i]
        ax.bar(np.arange(num_datasets), avg_values[:, i], yerr=sem_values[:, i], color=colors, capsize=5)
        ax.errorbar(0, real_avg_values[i], yerr=sem_values[0, i], fmt='o', color='red', markersize=8, capsize=5)
        ax.set_xticks(np.arange(num_datasets))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=14, fontweight='bold')  # Adjust font size and weight
        ax.set_title(plot_titles[i], fontsize=16, fontweight='bold')  # Adjust font size and weight
        ax.set_ylabel('Average', fontsize=16, fontweight='bold')  # Adjust font size and weight
        ax.axhline(real_avg_values[i], color='red', linestyle='--', linewidth=1)  # Add a line for real data average
        ax.yaxis.set_tick_params(labelsize=14)

    plt.tight_layout()
    plt.yticks(fontsize=14, fontweight='bold')
    plt.show()

################################################################

# Define a list of file names for both text and FASTA files
fasta_files = ['real_AMP.fasta', 'random_protein_sequences.fasta','FBGAN_above_0.8.fasta', 'FBGAN_kmers_above_0.8.fasta', 'FBGAN_ESM2above_0.8.fasta','AMPGAN_above_0.8.fasta', 'hydrAMP_above_0.8.fasta']
feature_names = ['Charge', 'pI', 'Aromaticity', 'HydrophobicityRatio']

# Create an empty list to store the DataFrames
dfs_generated = []

# Process each text file and its corresponding FASTA file
for fasta_file in fasta_files:
    df_generated = calculate_protein_descriptors_and_features(fasta_file)
    dfs_generated.append(df_generated)

features_datasets = []
for feature_name in feature_names:
    feature_datasets = [df_generated[feature_name] for df_generated in dfs_generated]
    features_datasets.append(feature_datasets)

# Define colors for each dataset
colors = ['blue', 'gray', 'pink','yellow','red', 'purple','green']
labels = ['Real data', 'Random data', 'FBGAN', 'FBGAN-kmers', 'FBGAN-ESM2', 'AMPGAN', 'HydrAMP']

plot_average_comparison(features_datasets, feature_names, labels, colors)

