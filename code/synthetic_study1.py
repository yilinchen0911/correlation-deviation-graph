from utility import fisher_transform
from SpatialGraph import SpatialGraph
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn import metrics
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
sns.set()

graph_construction_method_list = ['Unweighted', 'Pearson', 'Deviation']

number_of_nodes_for_simulation_number = 50
simulation_number_range = np.arange(5, 150, 20)
rho_for_simulation_number_range = 0.5
epsilon_for_simulation_number_range = 0.1

number_of_nodes_for_epsilon_range = 50
simulation_number_for_epsilon_range = 20
rho_for_epsilon_range = 0.5
epsilon_range = np.arange(0.05, 0.45, 0.05)

number_of_nodes_for_rho_range = 50
simulation_number_for_rho_range = 20
epsilon_for_rho_range = 0.1
rho_range = np.arange(0.1, 0.8, 0.1)


def generate_sythetic_graph_function(number_of_nodes_per_cluster, epsilon, expected_correlarion, simulation_number, number_of_cluster, method, seed):
    correlation_matrix = np.zeros(
        (number_of_cluster * number_of_nodes_per_cluster, number_of_cluster * number_of_nodes_per_cluster))
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            if i // number_of_nodes_per_cluster == j // number_of_nodes_per_cluster:
                correlation_matrix[i, j] = expected_correlarion + epsilon
            else:
                correlation_matrix[i, j] = expected_correlarion
            if i == j:
                correlation_matrix[i, j] = 1
    np.random.seed(seed)
    X = np.random.multivariate_normal(np.repeat(
        0, number_of_cluster * number_of_nodes_per_cluster), correlation_matrix, simulation_number)
    correlation_matrix_hat = np.corrcoef(X.T)
    file = open('graph/temp.txt', "w")
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            coff = correlation_matrix_hat[i, j]
            coff_cut = 1 if coff > expected_correlarion else 0
            deviation = (fisher_transform(
                coff) - fisher_transform(expected_correlarion)) * np.sqrt(simulation_number-3.0)
            if method == 'Deviation':
                content = deviation
            elif method == 'Unweighted':
                content = coff_cut
            elif method == 'Pearson':
                content = coff
            file.write(str(i) + ' ' + str(j) + ' ' +
                       str(round(content, 3)) + '\n')
    file.close()
    spatial_graph_obj = SpatialGraph(
        graph_path='graph/temp.txt', data_dict_path=None, map_limit=None)
    return spatial_graph_obj


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return 1.0 * np.sum(np.amax(contingency_matrix, axis=0)) / len(y_true)


def F_measure(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    S_k = np.sum(contingency_matrix, axis=0)
    S_khat = np.sum(contingency_matrix, axis=1)
    P_khat_k = 1.0 * contingency_matrix / S_k[None, :]
    R_khat_k = 1.0 * contingency_matrix / S_khat[:, None]
    F_khat_k = 2 * P_khat_k * R_khat_k / (P_khat_k + R_khat_k)
    F = np.average(np.nanmax(F_khat_k, axis=1), weights=S_khat)
    return F


def evaluate_single_graph(graph_obj, true_label):
    partition = graph_obj.partition
    predicted_label = [0] * sum([len(i) for i in partition])
    for i in range(len(partition)):
        for j in partition[i]:
            predicted_label[j] = i
    return F_measure(true_label, predicted_label)


def evluate_graphs(method, clustering_method, number_of_cluster, number_of_sample):
    if method == "simulation_number":
        true_label = np.repeat(np.arange(number_of_cluster),
                               number_of_nodes_for_simulation_number)
        results = {}
        for graph_construction_method in graph_construction_method_list:
            results_graph_construction = []
            for simulation_number in simulation_number_range:
                acc_list = []
                for seed in range(number_of_sample):
                    graph_obj = generate_sythetic_graph_function(number_of_nodes_for_simulation_number, epsilon_for_simulation_number_range,
                                                                 rho_for_simulation_number_range, simulation_number, number_of_cluster, graph_construction_method, seed)
                    if graph_construction_method == 'Deviation':
                        num_eig = 2
                    else:
                        num_eig = 2
                    graph_obj.clustering(
                        method=clustering_method, num_clust=number_of_cluster, num_eig=num_eig)
                    acc = evaluate_single_graph(graph_obj, true_label)
                    acc_list.append(acc)
                results_graph_construction.append(
                    (np.mean(acc_list), np.std(acc_list)))
            results[graph_construction_method] = copy.deepcopy(
                results_graph_construction)
    if method == "epsilon":
        true_label = np.repeat(np.arange(number_of_cluster),
                               number_of_nodes_for_epsilon_range)
        results = {}
        for graph_construction_method in graph_construction_method_list:
            results_graph_construction = []
            for epsilon in epsilon_range:
                acc_list = []
                for seed in range(number_of_sample):
                    graph_obj = generate_sythetic_graph_function(
                        number_of_nodes_for_epsilon_range, epsilon, rho_for_epsilon_range, simulation_number_for_epsilon_range, number_of_cluster, graph_construction_method, seed)
                    if graph_construction_method == 'Deviation':
                        num_eig = 2
                    else:
                        num_eig = 2
                    graph_obj.clustering(
                        method=clustering_method, num_clust=number_of_cluster, num_eig=num_eig)
                    acc = evaluate_single_graph(graph_obj, true_label)
                    acc_list.append(acc)
                results_graph_construction.append(
                    (np.mean(acc_list), np.std(acc_list)))
            results[graph_construction_method] = copy.deepcopy(
                results_graph_construction)
    if method == "rho":
        true_label = np.repeat(np.arange(number_of_cluster),
                               number_of_nodes_for_rho_range)
        results = {}
        for graph_construction_method in graph_construction_method_list:
            results_graph_construction = []
            for rho in rho_range:
                acc_list = []
                for seed in range(number_of_sample):
                    graph_obj = generate_sythetic_graph_function(
                        number_of_nodes_for_rho_range, epsilon_for_rho_range, rho, simulation_number_for_rho_range, number_of_cluster, graph_construction_method, seed)
                    if graph_construction_method == 'Deviation':
                        num_eig = 2
                    else:
                        num_eig = 2
                    graph_obj.clustering(
                        method=clustering_method, num_clust=number_of_cluster, num_eig=num_eig)
                    acc = evaluate_single_graph(graph_obj, true_label)
                    acc_list.append(acc)
                results_graph_construction.append(
                    (np.mean(acc_list), np.std(acc_list)))
            results[graph_construction_method] = copy.deepcopy(
                results_graph_construction)
    return results


def plot_accuracy(x, result, xlabel, fig_name):
    fig, ax = plt.subplots(figsize=(4, 4))
    for graph_construction_method in graph_construction_method_list:
        ax.errorbar(x, [i[0] for i in result[graph_construction_method]], yerr=[
                    i[1] for i in result[graph_construction_method]], label=graph_construction_method, uplims=True, lolims=True)
    ax.set(ylim=(0, 1.2))
    ax.set_ylabel('F')
    ax.legend()
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=300)


if __name__ == '__main__':
    results = evluate_graphs('simulation_number', 'Louvain', 2, 10)
    plot_accuracy(simulation_number_range, results, 'Number of Observations, m',
                  'figure/simulationNumber_louvain_new2.png')
    results = evluate_graphs('simulation_number', 'Spectral', 2, 10)
    plot_accuracy(simulation_number_range, results, 'Number of Observations, m',
                  'figure/simulationNumber_spectral_new2.png')
    results = evluate_graphs('epsilon', 'Louvain', 2, 10)
    plot_accuracy(epsilon_range, results, '$\\Delta$',
                  'figure/epsilon_louvain_new2.png')
    results = evluate_graphs('epsilon', 'Spectral', 2, 10)
    plot_accuracy(epsilon_range, results, '$\\Delta$',
                  'figure/epsilon_spectral_new2.png')
    results = evluate_graphs('rho', 'Louvain', 2, 10)
    plot_accuracy(rho_range, results, '$\\rho_0$',
                  'figure/rho_louvain_new2.png')
    results = evluate_graphs('rho', 'Spectral', 2, 10)
    plot_accuracy(rho_range, results, '$\\rho_0$',
                  'figure/rho_spectral_new2.png')
