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

### Train the classifiers and save the best model
To do so, run:  
- FBGAN classifier: `amp_predictor_pytorch.py`
- kmers-based classifer: `train_kmers_classifier.py`
- ESM2-based classifier: `train_MLP_classifier.py`
  
To run `train_MLP_classifier.py` first download `mean_embeddings_esm2_t12.csv` from [Google Drive](https://drive.google.com/drive/folders/1ZqWM7aBK1EmOc13uP7a4D03Llztb7uvO?usp=sharing). The expected output for the kmers-based and ESM2-based classifiers is the best model saved in a `.pth` format. The best model for FBGAN classifier should be stored in `checkpoint_FBGAN_classifier` folder. You can find the pre-trained classifiers utilized in this work on the [Google Drive](https://drive.google.com/drive/folders/1ZqWM7aBK1EmOc13uP7a4D03Llztb7uvO?usp=sharing).

### Train the generative models
For each model run the following:
- FBGAN: `wgan_gp_lang_gene_analyzer_FBGAN.py`
- FBGAN-kmers: `wgan_gp_lang_gene_FBGAN_kmers.py`
- FBGAN-ESM2: `wgan_gp_lang_gene_FBGAN_ESM2.py`
  
The expected output is a folder with checkpoints for each model. The optimal checkpoints utilized in this work for each model are provided on [Google Drive](https://drive.google.com/drive/folders/1ZqWM7aBK1EmOc13uP7a4D03Llztb7uvO?usp=sharing).

### Generate and select valid peptides
First, select the optimal model from the previous checkpoints based on the loss plots. Save them in a folder named `checkpoint_MODEL` where MODEL = {FBGAN, FBGAN-kmers, FBGAN-ESM2} and run:
- FBGAN: `generate_samples_FBGAN.py`
- FBGAN-kmers: `generate_samples_FBGAN_kmers.py`
- FBGAN-ESM2: `generate_samples_FBGAN_ESM2.py`
  
The expected output is a `.txt` file for each model with all the generated sequences. Then, for each output run `select_valid_peptides.py` to create a `.fasta` file that contains validly generated peptides.

### Evaluate the models
Use the code provided in the folder `evaluation`. The codes require the `.fasta` files created in the previous step.
- To plot the average physiochemical values you need to first run the [CAMPR4 server](https://camp.bicnirrh.res.in/predict/) and select the peptides with $P(\text{AMP}) \geq 0.8$.

### Reproduce the paper's results
* To reproduce the figures and results presented in the paper, please download the necessary files in `generated_peptides` folder from [Google Drive](https://drive.google.com/drive/folders/1ZqWM7aBK1EmOc13uP7a4D03Llztb7uvO?usp=sharing), extract them into the `evaluation` folder and run each Python script.

## Citation
If you use this code or data in your research, please cite our paper:

Zervou, M.A.; Doutsi, E.; Pantazis, Y.; Tsakalides, P. De Novo Antimicrobial Peptide Design with Feedback Generative Adversarial Networks. Int. J. Mol. Sci. 2024, 25, 5506. https://doi.org/10.3390/ijms25105506

BibTeX:
```bibtex
@Article{ijms25105506,
AUTHOR = {Zervou, Michaela Areti and Doutsi, Effrosyni and Pantazis, Yannis and Tsakalides, Panagiotis},
TITLE = {De Novo Antimicrobial Peptide Design with Feedback Generative Adversarial Networks},
JOURNAL = {International Journal of Molecular Sciences},
VOLUME = {25},
YEAR = {2024},
NUMBER = {10},
ARTICLE-NUMBER = {5506},
URL = {https://www.mdpi.com/1422-0067/25/10/5506},
ISSN = {1422-0067},
ABSTRACT = {Antimicrobial peptides (AMPs) are promising candidates for new antibiotics due to their broad-spectrum activity against pathogens and reduced susceptibility to resistance development. Deep-learning techniques, such as deep generative models, offer a promising avenue to expedite the discovery and optimization of AMPs. A remarkable example is the Feedback Generative Adversarial Network (FBGAN), a deep generative model that incorporates a classifier during its training phase. Our study aims to explore the impact of enhanced classifiers on the generative capabilities of FBGAN. To this end, we introduce two alternative classifiers for the FBGAN framework, both surpassing the accuracy of the original classifier. The first classifier utilizes the k-mers technique, while the second applies transfer learning from the large protein language model Evolutionary Scale Modeling 2 (ESM2). Integrating these classifiers into FBGAN not only yields notable performance enhancements compared to the original FBGAN but also enables the proposed generative models to achieve comparable or even superior performance to established methods such as AMPGAN and HydrAMP. This achievement underscores the effectiveness of leveraging advanced classifiers within the FBGAN framework, enhancing its computational robustness for AMP de novo design and making it comparable to existing literature.},
DOI = {10.3390/ijms25105506}
}
