import time
import os

from inverse_power_method import inverse_power_method, inverse_power_method_test
from deflation_method import deflation_method
from utils import *

import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from scipy.sparse import  linalg

np.random.seed(0)

circle_path = os.path.join('data', 'Circle.csv')
circle_df = pd.read_csv(circle_path, header=None, names=['x','y'])

sigma = 1.0
k = 5 #[5,10,20]

dataset = circle_df


adjacency_matrix = calculate_adjacency_matrix(dataset, sigma, k)
degree_matrix = calculate_diagonal_matrix(adjacency_matrix)
laplacian_matrix = calculate_laplacian_matrix(adjacency_matrix, degree_matrix)

# Create non_symmetric sparse matrix
n = 1000
density = 0.1
non_symmetric_data = np.random.rand(int(n * n * density))
non_symmetric_row_indices = np.random.randint(0, n, size=non_symmetric_data.shape[0])
non_symmetric_col_indices = np.random.randint(0, n, size=non_symmetric_data.shape[0])
non_symmetric_matrix = sp.coo_matrix((non_symmetric_data, (non_symmetric_row_indices, non_symmetric_col_indices)), shape=(n, n))


fig, ax = plt.subplots(1, 2, figsize=(12,5))
fig.subplots_adjust(left=None, bottom=None, right=None, 
                    top=0.82, wspace=0.40, hspace=None)

# Calculate exact smallest eigenvalue for symmetric matrix
exact_eigenvalues = linalg.eigsh(adjacency_matrix, which='SM')
smallest_eigenvalue = np.min(np.abs(exact_eigenvalues[0]))

# Add reference lines for smallest eigenvalue for symmetric matrix
ax[0].axhline(y=smallest_eigenvalue, linestyle='--', 
              c='r', label='Exact Eigenvalue Symm')

# Calculate exact smallest eigenvalue for non-symmetric matrix
exact_eigenvalues = linalg.eigs(non_symmetric_matrix, 3, sigma=0)
smallest_eigenvalue = np.min(np.abs(exact_eigenvalues[0]))

# Add reference lines for smallest eigenvalue for symmetric matrix
ax[1].axhline(y=smallest_eigenvalue, color='b', 
            linestyle='--', c='r', label='Exact Eigenvalue Non-Symm')

# Calculate approx smallest eigenvalue for symmetric matrix
eigenvalues, _, _, eigenvalues_iterations = inverse_power_method_test(
    laplacian_matrix, tol=1e-30)
ax[0].plot(range(1, len(eigenvalues_iterations) + 1), 
         eigenvalues_iterations, '-o', alpha=0.5, label='Approx Eigenvalue Symm')

# Calculate approx smallest eigenvalue for non-symmetric matrix
eigenvalues, _, _, eigenvalues_iterations = inverse_power_method_test(
        non_symmetric_matrix, tol=1e-30)
ax[1].plot(range(1, len(eigenvalues_iterations) + 1), 
         np.abs(eigenvalues_iterations), '-o', alpha=0.5, label='Approx Eigenvalue Non-symmetric')
# Plot
ax[0].set_title('Symmetric Matrix', fontweight='bold', fontsize=18)
ax[1].set_title('Non-Symmetric Matrix', fontweight='bold', fontsize=18)

ax[0].set_ylim([-.1, .1])
ax[1].set_ylim([0.2, 0.4])

ax[0].set_xlabel('Number of iterations', fontweight='bold', fontsize=18)
ax[0].set_ylabel('Eigenvalue', fontweight='bold', fontsize=18)

ax[1].set_xlabel('Number of iterations', fontweight='bold', fontsize=18)
ax[1].set_ylabel('Eigenvalue', fontweight='bold', fontsize=18)

ax[0].legend()
ax[1].legend()

fig.suptitle('Convergence of Eigenvalues', fontweight='bold', fontsize=18)

path_to_fig = os.path.join('figs', r'test_inverse_power_method_' + str(time.time()) + '.png')
fig.savefig(path_to_fig, dpi=600)
plt.show()