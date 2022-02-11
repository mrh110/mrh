# Simulating building the MLCM package
This simulation is to parctice the procedure for building a package on PyPi.

# MLCM creates a 2D Multi-Label Confusion Matrix
Please read the following paper for more information:
M. Heydarian, T. Doyle, and R. Samavi, MLCM: Multi-Label Confusion Matrix, IEEE Access, 2022

# An example on how to use MLCM packae:

import numpy as np
import sklearn.metrics as skm
from mlcm import mlcm

# creating "random true" and "random predicted" labels (multi-label); 
# 1000 samples of 5 classes.
number_of_samples = 1000
number_of_classes = 5
label_true = np.random.randint(2, size=(number_of_samples, number_of_classes))
label_pred = np.random.randint(2, size=(number_of_samples, number_of_classes))

conf_mat,normal_conf_mat = mlcm.cm(label_true,label_pred)
# Computing confusion matrix using 'MLCM'
print('\nRaw confusion Matrix:')
print(conf_mat)
print('\nNormalized confusion Matrix (%):')
print(normal_conf_mat)

# Scores using 'MLCM'
one_vs_rest = mlcm.stats(conf_mat)
# Scores using 'MLCM', not printing one-vs-rest matrices
one_vs_rest = mlcm.stats(conf_mat, False)
