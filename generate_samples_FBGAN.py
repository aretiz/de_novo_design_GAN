import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from sklearn.preprocessing import OneHotEncoder
import os, math, glob, argparse
from utils.torch_utils import *
from utils.utils import *
import matplotlib.pyplot as plt
import utils.language_helpers
plt.switch_backend('agg')
import numpy as np
from models import *

class WGAN_LangGP():
    def __init__(self, batch_size=64, lr=0.0001, num_epochs=150, seq_len = 156, data_dir='./data/dna_uniprot_under_50_reviewed.fasta', \
        run_name='test', hidden=512, d_steps = 10, max_examples=2000):
        self.preds_cutoff = 0.8
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = 1
        self.lamda = 10 #lambda
        self.checkpoint_dir = './checkpoint_FBGAN/' + run_name + "/"
        # self.sample_dir = './samples/' + run_name + "/"
        self.load_data(data_dir, max_examples) #max examples is size of discriminator training set
        # if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        # if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()

    def build_model(self):
        self.G = Generator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        if self.use_cuda:
            self.G.cuda()
        # print(self.G)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))

    def load_data(self, datadir, max_examples=1e6):
        self.data, self.charmap, self.inv_charmap = utils.language_helpers.load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
        self.labels = np.zeros(len(self.data)) #this marks at which epoch this data was added

    def load_model(self, directory='', iteration=None):
        '''
            Load model parameters from most recent epoch
        '''
        # print("Loading model from directory:", directory)  # Add this line for debugging
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth")
        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1  # file is not there
        if iteration is None:
            print("Loading most recently saved...")
            G_file = max(list_G, key=os.path.getctime)
        else:
            G_file = "G_weights_{}.pth".format(iteration)
        epoch_found = int((G_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found at {}!".format(epoch_found, directory))
        self.G.load_state_dict(torch.load(G_file))
        return epoch_found

    
    def generate_samples(self, num_samples=5000):
        """
        Generate samples using the trained generator.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            list: A list of generated sequences.
        """
        self.G.eval()  # Set generator to evaluation mode
        generated_sequences = []

        # Generate samples batch by batch
        with torch.no_grad():
            for _ in range(num_samples // self.batch_size):
                z_input = torch.randn(self.batch_size, 128)  # Generate random noise vectors
                if self.use_cuda:
                    z_input = z_input.cuda()
                samples = self.G(z_input)  # Generate sequences from random noise
                decoded_seqs = [decode_one_seq(seq, self.inv_charmap) for seq in samples.cpu().numpy()]
                generated_sequences.extend(decoded_seqs)

        return generated_sequences

def translate_dna_to_protein(dna_sequence):
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

def main():
    parser = argparse.ArgumentParser(description='Generate AMP.')
    parser.add_argument("--run_name", default="FBGAN", help="Name for output files")
    parser.add_argument("--load_dir", default="./checkpoint_FBGAN/", help="Load pretrained GAN checkpoints")
    parser.add_argument("--output_file", default="generated_samples_FBGAN.txt", help="Output file to save generated samples")
    args = parser.parse_args()
    model = WGAN_LangGP(run_name=args.run_name)

    # Load the generator model
    model.load_model(args.load_dir)

    # Generate 5000 samples
    generated_samples = model.generate_samples(num_samples=10000)
    
    # Write samples to a text file
    with open(args.output_file, "w") as f:
        for i, seq in enumerate(generated_samples):
            f.write(seq + '\n')
            # f.write(translate_dna_to_protein(seq) + '\n')

    print(f"Generated samples saved to {args.output_file}")

if __name__ == '__main__':
    main()
