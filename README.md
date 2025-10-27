<h1 align='center'>Continuum Transformers Perform In-Context Learning by Operator Gradient Descent</h1>

<div align='center'>
    <a href='https://abhiti23.github.io/' target='_blank'>Abhiti Mishra</a><sup>1</sup>&emsp;
    <a href='https://yashpatel5400.github.io/' target='_blank'>Yash Patel</a><sup>1</sup>&emsp;;
    <a href='https://www.ambujtewari.com/' target='_blank'>Ambuj Tewari</a><sup>2</sup>&emsp;
</div>

<div align='center'>
Department of Statistics, University of Michigan.
</div>

<p align='center'>
    <sup>1</sup>Equal contributions&emsp;
    <sup>2</sup>Senior investigator
</p>
<div align='center'>
    <a href='https://arxiv.org/abs/2505.17838'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>

## ⚒️ Automatic Pipeline
In-context learning for operator spaces

The implementation of the FANO operator upon which this work is based is thanks to the
code gratefully provided in: https://github.com/EdoardoCalvello/TransformerNeuralOperators/

# BLUP Experiments
To reproduce the BLUP experimental results, run the following
```
python blup_experiment.py --seed [seed]
```
Run this across many seed values to generate different trials. This will produces results in a 
results/ folder (as pkl files). To then visualize the results, run
```
python visualize.py
```
This will produce a "blup_final.png" file in the results/ directory.

## ⚖️ Disclaimer
This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.

## &#x1F4D2; Citation

If you find our work useful for your research, please consider citing the paper :

```
@article{mishra2025continuum,
  title={Continuum Transformers Perform In-Context Learning by Operator Gradient Descent},
  author={Mishra, Abhiti and Patel, Yash and Tewari, Ambuj},
  journal={arXiv preprint arXiv:2505.17838},
  year={2025}
}
```