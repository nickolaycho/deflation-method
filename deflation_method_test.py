#! free_hw_env\Scripts\python.exe

import time
import os

from inverse_power_method import inverse_power_method, inverse_power_method_test
from deflation_method import deflation_method, deflation_method_test
from utils import *

import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse import linalg

np.random.seed(0)

circle_path = os.path.join('data', 'Circle.csv')
spiral_path = os.path.join('data', 'Spiral.csv')

circle_df = pd.read_csv(circle_path, header=None, names=['x','y'])
spiral_df = pd.read_csv(spiral_path, header=None, 
                        names=['x','y', 'cluster'])
spiral_df = spiral_df[["x","y"]]
sigma = 1.0
k_values = [5,20]

datasets = [circle_df, spiral_df]

fig, ax = plt.subplots(len(k_values), len(datasets), figsize=(12,6))

fig.subplots_adjust(left=None, bottom=None, right=None, 
                    top=0.82, wspace=0.40, hspace=None)

for i, dataset in enumerate(datasets):
    for j,k in enumerate(k_values):
        adjacency_matrix = calculate_adjacency_matrix(dataset, sigma, k)
        degree_matrix = calculate_diagonal_matrix(adjacency_matrix)
        laplacian_matrix = calculate_laplacian_matrix(adjacency_matrix, degree_matrix)

        eigenvalues, _ = deflation_method_test(A=laplacian_matrix, 
                            M=10, niterations_max=100, tol=1e-25)

        theoric_eigenvalues = linalg.eigs(laplacian_matrix, which='SM', k=10)[0]
        theoric_eigenvalues = np.sort(theoric_eigenvalues)

        ax[j,i].plot(theoric_eigenvalues, '-o', label="theoretical eigenvalues")
        ax[j,i].legend()

        ax[j,i].plot(np.abs(eigenvalues), '--+', label='deflation power method')
        
        # Plot settings
        #ax[j,i].set_title('Symmetric Matrix', fontweight='bold', fontsize=18)
        #ax[1].set_title('Non-Symmetric Matrix', fontweight='bold', fontsize=18)

        ax[j,i].set_ylabel(r'|$\lambda_i$|', fontweight='bold', fontsize=18)

        ax[j,i].legend()
        ax[j,i].set_xticks([s for s in range(10) ]) 
        ax[j,i].set_xticklabels([f'$\lambda_{s+1}$' for s in range(10)], fontsize=12)

fig.suptitle('Deflation Power Method Assessment', fontweight='bold', fontsize=18)

path_to_fig = os.path.join('figs', r'test_deflation_method_' + str(time.time()) + '.png')
fig.savefig(path_to_fig, dpi=600)

fig.tight_layout()
plt.show()

