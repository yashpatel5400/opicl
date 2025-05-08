# opicl
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