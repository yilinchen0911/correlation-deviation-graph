import numpy as np
import matplotlib.pyplot as plt
from synthetic_study1 import generate_sythetic_graph_function
from utility import compute_eigen_gaps

number_of_nodes_for_simulation_number = 50
simulation_number_range = np.arange(5, 150, 20)
rho_for_simulation_number_range = 0.5
epsilon_for_simulation_number_range = 0.1

number_of_cluster = [2, 3, 4, 5, 6]
number_of_eigenval = 10
number_of_simulation = 10
eigengap = np.zeros(
    (len(number_of_cluster), number_of_eigenval, number_of_simulation))


def main():
    for i in range(len(number_of_cluster)):
        for j in range(number_of_simulation):
            deviation_graph = generate_sythetic_graph_function(number_of_nodes_for_simulation_number,
                                                               epsilon_for_simulation_number_range,
                                                               rho_for_simulation_number_range, 100, number_of_cluster[
                                                                   i],
                                                               'Deviation', j)
            A = deviation_graph.get_adjacency_matrix()
            eigengap[i, :, j] = compute_eigen_gaps(A, number_of_eigenval)
    eigen_diff_mean = np.mean(eigengap, axis=2)
    eigen_diff_std = np.std(eigengap, axis=2)
    x = np.arange(1, number_of_eigenval + 1)
    for i in range(len(number_of_cluster)):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.errorbar(x[1:number_of_eigenval], eigen_diff_mean[i, 1:number_of_eigenval],
                    yerr=eigen_diff_std[i, 1:number_of_eigenval], uplims=True, lolims=True, fmt='o')
        ax.set_ylabel('$\delta_k$')
        ax.set_xlabel('$k$')
        ax.set_xticks(x[1:number_of_eigenval])
        ax.errorbar(x[number_of_cluster[i] - 1], eigen_diff_mean[i, number_of_cluster[i] - 1], yerr=eigen_diff_std[i, number_of_cluster[i] - 1],
                    uplims=True, lolims=True, color='red', fmt='o')
        ax.set_ylim([0, 30])
        fig.tight_layout()
        fig.savefig('figure/eigen_gap_' + 'K = ' +
                    str(number_of_cluster[i]), dpi=300, bbox_inches = 'tight', pad_inches = 0)


if __name__ == '__main__':
    main()
