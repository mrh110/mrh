# Simulating building the MLCM package
This simulation is to parctice the procedure for building a package on PyPi.

# MLCM creates a 2D Multi-Label Confusion Matrix
Please read the following paper for more information:
M. Heydarian, T. Doyle, and R. Samavi, MLCM: Multi-Label Confusion Matrix, IEEE Access, 2022

# An example on how to use MLCM packae:

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# How to use \"Multi-Label Confusion Matrix\" (MLCM) algorithm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Please read the following paper for more information:\\\n",
    "M. Heydarian, T. Doyle, and R. Samavi, MLCM: Multi-Label Confusion Matrix, \n",
    "IEEE Access, 2022"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import sklearn.metrics as skm\n",
    "from mlcm import mlcm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creating random True and Predicted labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# creating \"random true\" and \"random predicted\" labels (multi-label); \n",
    "# 1000 samples of 5 classes.\n",
    "number_of_samples = 1000\n",
    "number_of_classes = 5\n",
    "label_true = np.random.randint(2, size=(number_of_samples, number_of_classes))\n",
    "label_pred = np.random.randint(2, size=(number_of_samples, number_of_classes))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "examples of True labels\n",
      " [[0 1 0 0 1]\n",
      " [1 0 1 0 1]\n",
      " [0 1 1 0 1]\n",
      " [1 0 0 0 1]]\n",
      "examples of Predicted labels\n",
      " [[0 1 0 1 0]\n",
      " [0 0 0 0 1]\n",
      " [0 1 0 0 0]\n",
      " [0 0 0 0 1]]\n"
     ]
    }
   ],
   "source": [
    "print('examples of True labels\\n',label_true[:4])\n",
    "print('examples of Predicted labels\\n',label_pred[:4])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Computing confusion matrix using 'MLCM\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MLCM has one extra row (NTL) and one extra column (NPL).        \n",
      "Please read the following paper for more information:\n",
      "        Heydarian et al., MLCM: Multi-Label Confusion Matrix, IEEE Access,2022        \n",
      "To skip this message, please add parameter \"print_note=False\"\n",
      "        e.g., conf_mat,normal_conf_mat = mlcm.cm(label_true,label_pred,False)\n",
      "\n",
      "Raw confusion Matrix:\n",
      "[[284  96  78  94 118  95]\n",
      " [ 75 231  89  92 103  85]\n",
      " [ 75  94 247  86 115  97]\n",
      " [ 73 104 103 251 109  69]\n",
      " [ 78  90  88  94 266  89]\n",
      " [  9  12  14  10  14   1]]\n",
      "\n",
      "Normalized confusion Matrix (%):\n",
      "[[37. 13. 10. 12. 15. 12.]\n",
      " [11. 34. 13. 14. 15. 13.]\n",
      " [11. 13. 35. 12. 16. 14.]\n",
      " [10. 15. 15. 35. 15. 10.]\n",
      " [11. 13. 12. 13. 38. 13.]\n",
      " [15. 20. 23. 17. 23.  2.]]\n"
     ]
    }
   ],
   "source": [
    "conf_mat,normal_conf_mat = mlcm.cm(label_true,label_pred)\n",
    "\n",
    "print('\\nRaw confusion Matrix:')\n",
    "print(conf_mat)\n",
    "print('\\nNormalized confusion Matrix (%):')\n",
    "print(normal_conf_mat) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Scores using MLCM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[ 996  310]\n",
      "  [ 481  284]]\n",
      "\n",
      " [[1049  396]\n",
      "  [ 444  231]]\n",
      "\n",
      " [[1033  372]\n",
      "  [ 467  247]]\n",
      "\n",
      " [[1029  376]\n",
      "  [ 458  251]]\n",
      "\n",
      " [[1014  459]\n",
      "  [ 439  266]]\n",
      "\n",
      " [[1279  435]\n",
      "  [  59    1]]]\n",
      "\n",
      "       class#     precision        recall      f1-score        weight\n",
      "\n",
      "            0          0.48          0.37          0.42          765\n",
      "            1          0.37          0.34          0.35          675\n",
      "            2          0.40          0.35          0.37          714\n",
      "            3          0.40          0.35          0.38          709\n",
      "            4          0.37          0.38          0.37          705\n",
      "          NTL          0.00          0.02          0.00          60\n",
      "\n",
      "    micro avg          0.35          0.35          0.35          3628\n",
      "    macro avg          0.34          0.30          0.32          3628\n",
      " weighted avg          0.40          0.35          0.37          3628\n"
     ]
    }
   ],
   "source": [
    "one_vs_rest = mlcm.stats(conf_mat)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "       class#     precision        recall      f1-score        weight\n",
      "\n",
      "            0          0.48          0.37          0.42          765\n",
      "            1          0.37          0.34          0.35          675\n",
      "            2          0.40          0.35          0.37          714\n",
      "            3          0.40          0.35          0.38          709\n",
      "            4          0.37          0.38          0.37          705\n",
      "          NTL          0.00          0.02          0.00          60\n",
      "\n",
      "    micro avg          0.35          0.35          0.35          3628\n",
      "    macro avg          0.34          0.30          0.32          3628\n",
      " weighted avg          0.40          0.35          0.37          3628\n"
     ]
    }
   ],
   "source": [
    "one_vs_rest = mlcm.stats(conf_mat, False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " [[[ 996  310]\n",
      "  [ 481  284]]\n",
      "\n",
      " [[1049  396]\n",
      "  [ 444  231]]\n",
      "\n",
      " [[1033  372]\n",
      "  [ 467  247]]\n",
      "\n",
      " [[1029  376]\n",
      "  [ 458  251]]\n",
      "\n",
      " [[1014  459]\n",
      "  [ 439  266]]\n",
      "\n",
      " [[1279  435]\n",
      "  [  59    1]]]\n"
     ]
    }
   ],
   "source": [
    "print('\\n',one_vs_rest)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Scores using scikit-learn library"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[241 215]\n",
      "  [260 284]]\n",
      "\n",
      " [[275 251]\n",
      "  [243 231]]\n",
      "\n",
      " [[260 229]\n",
      "  [264 247]]\n",
      "\n",
      " [[297 231]\n",
      "  [221 251]]\n",
      "\n",
      " [[229 269]\n",
      "  [236 266]]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.57      0.52      0.54       544\n",
      "           1       0.48      0.49      0.48       474\n",
      "           2       0.52      0.48      0.50       511\n",
      "           3       0.52      0.53      0.53       472\n",
      "           4       0.50      0.53      0.51       502\n",
      "\n",
      "   micro avg       0.52      0.51      0.51      2503\n",
      "   macro avg       0.52      0.51      0.51      2503\n",
      "weighted avg       0.52      0.51      0.51      2503\n",
      " samples avg       0.51      0.50      0.46      2503\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/lib/python3/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n",
      "/usr/lib/python3/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in samples with no true labels. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n"
     ]
    }
   ],
   "source": [
    "cm = skm.multilabel_confusion_matrix(label_true,label_pred)\n",
    "print(cm)\n",
    "print(skm.classification_report(label_true,label_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
