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
- Step A: `Train the classifiers.`
  
- Step B: `Description of step B.`
- Step C: `Description of step C.`
