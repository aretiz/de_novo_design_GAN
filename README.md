# De Novo Antimicrobial Peptide Design with Feedback Generative Adversarial Networks

 This study aims to explore the impact of enhanced classifiers on the generative capabilities
of FBGAN. To this end, we introduce two alternative classifiers for the FBGAN framework, both
surpassing the accuracy of the original classifier. The first classifier utilizes the k-mers technique,
while the second applies transfer learning from the large protein language model Evolutionary Scale
Modeling 2 (ESM2). Integrating these classifiers into FBGAN not only yields notable performance
enhancements compared to the original FBGAN but also enables the proposed generative models
to achieve comparable or even superior performance to established methods such as AMPGAN
and HydrAMP. This achievement underscores the effectiveness of leveraging advanced classifiers
within the FBGAN framework, enhancing its computational robustness for AMP de novo design and
making it comparable to existing literature.

## Install the dependencies
The code is tested under Windows running python 3.8. All required packages are enclosed in `requirements.txt`. Run:
```bash
pip install -r requirements.txt
```
## Peptide generation
To run this project, follow these steps:

# Train the classifiers and save the best model
To do so, run:  
- FBGAN: `amp_predictor_pytorch.py`
- FBGAN-kmers: `train_kmers_classifier.py`
- FBGAN-ESM2: `train_MLP_classifier.py`
  
The expected output is the best model saved in a `.pth` format.  

# Train the generative models
For each model run the following:
- FBGAN: `wgan_gp_lang_gene_analyzer_FBGAN.py`
- FBGAN-kmers: `wgan_gp_lang_gene_FBGAN_kmers.py`
- FBGAN-ESM2: `wgan_gp_lang_gene_FBGAN_ESM2.py`
  
The expected output is the folder with checkpoints for each model.

# Generate and select valid peptides
First run:
- FBGAN: `generate_samples_FBGAN.py`
- FBGAN-kmers: `generate_samples_FBGAN_kmers.py`
- FBGAN-ESM2: `generate_samples_FBGAN_ESM2.py`
  
The expected output is a `.txt` file for each model with all the generated sequences. Then,  for each output run `select_valid_peptides.py` to create a `.fasta` file that contains validly generated peptides.

# Evaluate the models
Use the code provided in the folder `evaluation`. The codes require the `.fasta` files created in the previous step.
- To plot the average physiochemical values you need to first run the `[CAMPR4 server] (https://camp.bicnirrh.res.in/predict/)` and select the peptides with `\( P(\text{AMP}) \geq 0.8 \)`.
