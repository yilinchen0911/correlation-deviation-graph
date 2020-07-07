from synthetic_study1 import purity_score, F_measure, plot_accuracy, evaluate_single_graph
from utility import fisher_transform
from SpatialGraph import SpatialGraph
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.covariance import graphical_lasso
import pandas as pd
import os
from sklearn import metrics
import copy
from utility import *
import seaborn as sns
sns.set()

graph_construction_method_list = ['Unweighted', 'Pearson', 'Deviation']

simulation_number_range = np.arange(5, 150, 20)
box1 = (3, 6, 2, 5)
box2 = (3, 6, 8, 11)
alpha_for_simulation_number_range = -0.3
epsilon_for_simulation_number_range = 0.1


def initialize_node_location(row, col):
    node_location = np.zeros((row * col, 2))
    for i in range(row):
        for j in range(col):
            node_location[i * col + j, :] = [i, j]
    return node_location


def initialize_node_label(row, col, box1, box2):
    node_label = np.zeros(row * col)
    for i in range(row):
        for j in range(col):
            if i >= box1[0] - 1 and i < box1[1] and j >= box1[2] - 1 and j < box1[3]:
                node_label[i * col + j] = 1
            if i >= box2[0] - 1 and i < box2[1] and j >= box2[2] - 1 and j < box2[3]:
                node_label[i * col + j] = 2
    return node_label


def get_distance_matrix(node_location):
    distance_matrix = np.zeros(
        (node_location.shape[0], node_location.shape[0]))
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance_matrix[i, j] = np.sqrt(
                np.sum((node_location[i, :] - node_location[j, :]) ** 2))
    return distance_matrix


def get_epsilon_matrix2(node_label, epsilon):
    epsilon_matrix = np.zeros((len(node_label), len(node_label)))
    for i in range(epsilon_matrix.shape[0]):
        for j in range(i + 1, epsilon_matrix.shape[1]):
            if node_label[i] == 1 and node_label[j] == 1:
                epsilon_matrix[i, j] = epsilon
            if node_label[i] == 2 and node_label[j] == 2:
                epsilon_matrix[i, j] = epsilon
            if node_label[i] != node_label[j]:
                epsilon_matrix[i, j] = -epsilon
            epsilon_matrix[j, i] = epsilon_matrix[i, j]
    return epsilon_matrix


def get_epsilon_matrix(node_label, epsilon):
    epsilon_matrix = np.zeros((len(node_label), len(node_label)))
    for i in range(epsilon_matrix.shape[0]):
        for j in range(i + 1, epsilon_matrix.shape[1]):
            epsilon_matrix[i,
                           j] = epsilon if node_label[i] == node_label[j] else 0
            epsilon_matrix[j, i] = epsilon_matrix[i, j]
    return epsilon_matrix


def initialize_convariance_matrix(node_location, node_label, alpha, epsilon):
    distance_matrix = get_distance_matrix(node_location)
    epsilon_matrix = get_epsilon_matrix(node_label, epsilon)
    convariance_matrix = np.exp(alpha * distance_matrix) + epsilon_matrix
    return convariance_matrix


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def generate_sythetic_graph_function(correlation_matrix, expected_correlarion_matrix, simulation_number, method, seed):
    np.random.seed(seed)
    X = np.random.multivariate_normal(np.repeat(
        0, correlation_matrix.shape[0]), correlation_matrix, simulation_number)
    correlation_matrix_hat = np.corrcoef(X.T)
    file = open('graph/temp.txt', "w")
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            coff = correlation_matrix_hat[i, j]
            coff_cut = 1 if coff > expected_correlarion_matrix[i, j] else 0
            deviation = (fisher_transform(coff) - fisher_transform(
                expected_correlarion_matrix[i, j])) * np.sqrt(simulation_number-3.0)
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


def evluate_graphs(correlation_matrix, expected_correlarion_matrix, method, clustering_method, number_of_cluster, number_of_sample):
    true_label = node_label
    if method == "simulation_number":
        results = {}
        for graph_construction_method in graph_construction_method_list:
            results_graph_construction = []
            for simulation_number in simulation_number_range:
                acc_list = []
                for seed in range(number_of_sample):
                    graph_obj = generate_sythetic_graph_function(
                        correlation_matrix, expected_correlarion_matrix, simulation_number, graph_construction_method, seed)
                    if graph_construction_method == 'Deviation':
                        num_eig = 3
                    else:
                        # First eigenvector is constant, here ignore this, all use three
                        num_eig = 3
                    graph_obj.clustering(
                        method=clustering_method, num_clust=number_of_cluster, num_eig=num_eig)
                    acc = evaluate_single_graph(graph_obj, true_label)
                    acc_list.append(acc)
                results_graph_construction.append(
                    (np.mean(acc_list), np.std(acc_list)))
            results[graph_construction_method] = copy.deepcopy(
                results_graph_construction)
    return results


def visualize_synthetic_data(node_location, node_label, file_name, order=[0, 1, 2]):
    fig, ax = plt.subplots(figsize=(12, 8), linewidth=5, edgecolor="black")
    maker_list = ['^', 'o', 's']
    for i in range(len(order)):
        ax.scatter(node_location[node_label == order[i], 1],
                   node_location[node_label == order[i], 0], s=400, marker=maker_list[i])
    plt.axis('off')
    fig.savefig(file_name, dpi=300)


def get_graph_node_label(graph_obj):
    partition = graph_obj.partition
    predicted_label = [0] * sum([len(i) for i in partition])
    for i in range(len(partition)):
        for j in partition[i]:
            predicted_label[j] = i
    return np.array(predicted_label)


if __name__ == '__main__':
    node_location = initialize_node_location(8, 12)
    node_label = initialize_node_label(8, 12, box1, box2)
    distance_matrix = get_distance_matrix(node_location)
    expected_correlarion_matrix = np.exp(
        alpha_for_simulation_number_range * distance_matrix)
    correlation_matrix = initialize_convariance_matrix(
        node_location, node_label, alpha_for_simulation_number_range, epsilon_for_simulation_number_range)
    assert is_pos_def(correlation_matrix)

    visualize_synthetic_data(node_location, node_label,
                             'figure/2True_synthetic_detected.png')

    results = evluate_graphs(
        correlation_matrix, expected_correlarion_matrix, 'simulation_number', 'Spectral', 3, 10)
    plot_accuracy(simulation_number_range, results, 'Number of Observations, m',
                  'figure/2simulationNumber_spectral_new.png')

    graph_obj = generate_sythetic_graph_function(
        correlation_matrix, expected_correlarion_matrix, 145, "Deviation", 2)
    graph_obj.clustering(method='Spectral', num_clust=3, num_eig=3)
    visualize_synthetic_data(node_location, get_graph_node_label(
        graph_obj), 'figure/2Deviation_synthetic_detected.png', order=[1, 2, 0])
    graph_obj = generate_sythetic_graph_function(
        correlation_matrix, expected_correlarion_matrix, 145, "Unweighted", 2)
    graph_obj.clustering(method='Spectral', num_clust=3, num_eig=3)
    visualize_synthetic_data(node_location, get_graph_node_label(
        graph_obj), 'figure/2Unweighted_synthetic__detected.png', order=[1, 2, 0])
    graph_obj = generate_sythetic_graph_function(
        correlation_matrix, expected_correlarion_matrix, 145, "Pearson", 2)
    graph_obj.clustering(method='Spectral', num_clust=3, num_eig=3)
    visualize_synthetic_data(node_location, get_graph_node_label(
        graph_obj), 'figure/2Pearson_synthetic__detected.png', order=[1, 2, 0])
