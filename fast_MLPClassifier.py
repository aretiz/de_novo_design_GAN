import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import logging
import re

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
    'TAA': '', 'TAG': '', 'TGA': '', # '***': '#', # Stop codons or placeholders ('X')
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

class MLPclassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPclassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2) # 2 refers to the number of classes
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class AMPClassifier():
    def __init__(self, hidden_dim=128, epochs=30, batch_size=16):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_epochs = epochs

    def get_protein_mean_embeddings(self, protein_sequences): #, max_length):
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

        model_name="facebook/esm2_t12_35M_UR50D"

        # Load the pre-trained model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Tokenize the protein sequence
        tokens = tokenizer(protein_sequences, return_tensors="pt", padding=True)

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**tokens)

        # Extract embeddings from the model's output
        embeddings = outputs.last_hidden_state  

        # Calculate the mean along the 'sequence_length' dimension
        mean_embeddings = torch.mean(embeddings, dim=1)

        return mean_embeddings

    def translate_dna_to_protein(self, dna_sequence, translation_table):
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
    
    def predict_model(self, dna_seqs):
        prot_seqs = []
        for i in range(len(dna_seqs)):
            seq = re.sub(r'P*$', '', dna_seqs[i])
            prot = self.translate_dna_to_protein(seq, translation_table)
            prot_seqs.append(prot)
        
        emb = self.get_protein_mean_embeddings(prot_seqs) 
        input_size = len(emb[0])

        # define the model
        self.model = MLPclassifier(input_size, self.hidden_dim)
        self.model.to(self.device)

        self.model.load_state_dict(torch.load('./best_model_ESM2.pth'))
        
        loader = DataLoader(emb, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                outputs = self.model(x)
                probabilities = torch.softmax(outputs, dim=1)  # Calculate class probabilities
                positive_class_probabilities = probabilities[:, 1].cpu().numpy()
                predictions.extend(positive_class_probabilities.tolist())
        return np.array(predictions).reshape(-1, 1)
