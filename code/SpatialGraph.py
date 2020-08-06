from EmpiricalSpatialCorrelationModel import EmpiricalSpatialCorrelationModel
from utility import load_obj, compute_eigen_gaps, get_correlation_coefficient, get_distance, mean_in_each_bin
from SignedSpectral import SignedSpectral
from SignedLouvain import SignedLouvain
import geopy.distance
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


class SpatialGraph(object):
    def __init__(self, graph_path="graph.txt", data_dict_path="dataDict",
                 method="Louvain", map_limit={'lat_0': -41.3, 'lon_0': 174.8, 'width': 4E4, 'height': 4E4}):
        self.method = method
        self.data_dict = load_obj(data_dict_path)
        self.edge_dict = {}
        nodes = []
        with open(graph_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                node_id1 = int(row[0])
                node_id2 = int(row[1])
                edge = float(row[2])
                self.edge_dict[frozenset([node_id1, node_id2])] = edge
                nodes.append(node_id1)
                nodes.append(node_id2)
        self.nodes = np.unique(nodes)
        self.map_limit = map_limit
        self.graph_path = graph_path

    def clustering(self, method="Spectral", sign="positive", num_clust=10, num_eig=5):
        if method == "Louvain":
            pyl = SignedLouvain.from_file(self.graph_path, sign)
            partition_result, q = pyl.apply_method()
            self.partition = [self.nodes[i] for i in partition_result]
            self.q = q
        elif method == "Spectral":
            spc = SignedSpectral(self.graph_path)
            partition_result = spc.apply_method(num_clust, num_eig)
            self.partition = [self.nodes[i] for i in partition_result]
            self.q = None
        elif method == "First Eigenvector":
            spc = signedSpectral(self.graph_path)
            partition_result = spc.firstEigenVector()
            self.partition = [self.nodes[i] for i in partition_result]
            self.q = None
        elif method == "Second Eigenvector":
            spc = signedSpectral(self.graph_path)
            partition_result = spc.second_eigenVector()
            self.partition = [self.nodes[i] for i in partition_result]
            self.q = None

    def visualze_graph(self, figure_file_name="test.png"):
        # 1. Draw the map background
        fig = plt.figure(figsize=(8, 8))
        m = Basemap(projection='lcc', resolution='h',
                    lat_0=self.map_limit['lat_0'], lon_0=self.map_limit['lon_0'],
                    width=self.map_limit['width'], height=self.map_limit['height'])
        m.shadedrelief()
        m.drawcoastlines(color='gray')
        m.drawcountries(color='gray')
        m.drawstates(color='gray')
        cmap = plt.cm.seismic
        norm = matplotlib.colors.Normalize(vmin=-6, vmax=6)
        key_list = list(self.edge_dict.keys())
        for i in range(len(key_list)):
            edge = self.edge_dict[key_list[i]]
            node_pair = tuple(key_list[i])
            node1 = node_pair[0]
            node2 = node_pair[1]
            if True:
                lat1 = self.data_dict['StationLoc'][np.where(
                    self.data_dict['siteID'] == node1)[0][0]][1]
                lon1 = self.data_dict['StationLoc'][np.where(
                    self.data_dict['siteID'] == node1)[0][0]][0]
                lat2 = self.data_dict['StationLoc'][np.where(
                    self.data_dict['siteID'] == node2)[0][0]][1]
                lon2 = self.data_dict['StationLoc'][np.where(
                    self.data_dict['siteID'] == node2)[0][0]][0]
                lons = [lon1, lon2]
                lats = [lat1, lat2]
                x, y = m(lons, lats)
                m.plot(x, y, marker=None, color=cmap(norm(edge)), alpha=0.5)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # fig.colorbar(sm)
        fig.savefig(figure_file_name)
        plt.close(fig)

    def visualze_community_location(self, markerSize=10, figure_file_name="test.png"):
        # 1. Draw the map background
        fig = plt.figure(figsize=(8, 8))
        m = Basemap(projection='lcc', resolution='h',
                    lat_0=self.map_limit['lat_0'], lon_0=self.map_limit['lon_0'],
                    width=self.map_limit['width'], height=self.map_limit['height'])
        m.shadedrelief()
        m.drawcoastlines(color='gray')
        m.drawcountries(color='gray')
        m.drawstates(color='gray')
        cmap = plt.cm.seismic
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(self.partition))
        l = 0
        key_list = list(self.edge_dict.keys())
        maker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4',
                      '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']

        node_community = [[j for j, lst in enumerate(
            self.partition) if node in lst][0] for node in self.nodes]
        node_community_size = np.array(
            [len(self.partition[community]) for community in node_community])
        node_index = np.lexsort((node_community, -node_community_size))
        node_community_size_unique = np.sort(
            [len(k) for k in self.partition])[::-1]

        for i in range(len(self.nodes)):
            lat1 = self.data_dict['StationLoc'][np.where(
                self.data_dict['siteID'] == self.nodes[i])[0][0]][1]
            lon1 = self.data_dict['StationLoc'][np.where(
                self.data_dict['siteID'] == self.nodes[i])[0][0]][0]
            node1_community = [j for j, lst in enumerate(
                self.partition) if self.nodes[i] in lst][0]
            x, y = m(lon1, lat1)
            #m.scatter(x, y, s = markerSize, marker = maker_list[node1_community % len(maker_list)], color = cmap(norm(node1_community)), label = str(node1_community))
            community_sorted_by_size = next(m for m, val in enumerate(node_community_size_unique) if sum(
                node_community_size_unique[:(m+1)]) > np.where(node_index == i)[0][0])
            plt.annotate(str(community_sorted_by_size+1), xy=(x, y))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        handles, labels = fig.gca().get_legend_handles_labels()
        index = np.argsort([np.int(x) for x in labels])
        by_label = OrderedDict(
            zip(np.array(labels)[index], np.array(handles)[index]))
        #fig.legend(by_label.values(), by_label.keys(), loc = "center right", title = "Label")
        fig.savefig(figure_file_name)
        plt.close(fig)

    def project_to_line(self, x, origin=[-119.12872, 34.32412], reference=[-117.42888, 33.5324]):
        a = np.array(x) - np.array(origin)
        b = np.array(reference) - np.array(origin)
        a_l2norm = np.sqrt(np.sum(a ** 2))
        b_l2norm = np.sqrt(np.sum(b ** 2))
        cos = np.sum(a * b) / a_l2norm / b_l2norm
        coords_1 = (origin[1], origin[0])
        coords_2 = (x[1], x[0])
        distance = geopy.distance.vincenty(coords_1, coords_2).km
        return distance * cos

    def visualize_community_by_z10(self, file_name="community_by_z10.png"):
        cmap = plt.cm.seismic
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(self.partition))
        z10_dict = load_obj('z10_dict')
        fig, ax = plt.subplots(figsize=(12, 8), linewidth=5, edgecolor="black")
        maker_list = ['*', 'o', 'D', 's', '^']
        for i in range(len(self.partition)):
            z10 = [z10_dict[j] for j in self.partition[i]]
            lat = [self.data_dict['StationLoc'][np.where(self.data_dict['siteID'] == k)[
                0][0]][1] for k in self.partition[i]]
            lon = [self.data_dict['StationLoc'][np.where(self.data_dict['siteID'] == l)[
                0][0]][0] for l in self.partition[i]]
            x = [self.project_to_line([lon[m], lat[m]])
                 for m in range(len(lat))]
            ax.scatter(x, z10, s=40, marker=maker_list[i % len(
                maker_list)], color=cmap(norm(i)))
        ax.set_xlabel('Along strike direction [km]', fontsize=20)
        ax.set_ylabel('$Z_{1.0}$ [km]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.get_xaxis().set_ticks([0, 50, 100, 150, 200])
        ax.set_xlim([0, 200])
        fig.savefig(file_name, dpi=300)

    def get_community_location(self):
        cmap = plt.cm.seismic
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(self.partition))
        #maker_list = ['-', 'a','b','c','d','e','g','h','i','n','r','s','t','w','x']
        maker_list = ['a', 'c', 'd', 's', 't']
        results = []
        for i in range(len(self.nodes)):
            lat1 = self.data_dict['StationLoc'][np.where(
                self.data_dict['siteID'] == self.nodes[i])[0][0]][1]
            lon1 = self.data_dict['StationLoc'][np.where(
                self.data_dict['siteID'] == self.nodes[i])[0][0]][0]
            node1_community = [j for j, lst in enumerate(
                self.partition) if self.nodes[i] in lst][0]
            color = '/'.join(np.array([str(int(k * 255))
                                       for k in cmap(norm(node1_community))])[0:3])
            results.append(
                (lon1, lat1, maker_list[node1_community % len(maker_list)], color))
        return results

    def write_community_location_file(self, command_file, station_file_dir, path):
        results = self.get_community_location()
        command_file = open("/".join([path, command_file]), "w")
        for i in range(len(results)):
            content = "gmt psxy {}/{} -J$proj -R$region -G{} -S{}0.1i -O -K >> stationMap.ps".format(
                station_file_dir, str(i) + '.xy', results[i][3], results[i][2])
            command_file.write(content + '\n')
            station_file = open(
                "/".join([path, station_file_dir, str(i) + '.xy']), "w")
            station_file.write(str(results[i][0]) + ' ' + str(results[i][1]))
            station_file.close()
        command_file.close()

    def get_adjacency_matrix(self):
        A = np.zeros([len(self.nodes), len(self.nodes)])
        for edge in self.edge_dict.keys():
            EI = list(edge)
            node1 = EI[0]
            node2 = EI[1]
            rowIndex = np.where(self.nodes == node1)[0][0]
            colIndex = np.where(self.nodes == node2)[0][0]
            A[rowIndex, colIndex] = self.edge_dict[frozenset([node1, node2])]
            A[colIndex, rowIndex] = self.edge_dict[frozenset([node1, node2])]
        return A

    def read_z10(self, z10_file):
        z10_file_matrix = np.loadtxt(z10_file)
        z10_dict = {}
        for i in range(len(self.nodes)):
            Lat = self.data_dict['StationLoc'][np.where(
                self.data_dict['siteID'] == self.nodes[i])[0][0]][1]
            Lon = self.data_dict['StationLoc'][np.where(
                self.data_dict['siteID'] == self.nodes[i])[0][0]][0]
            dist_approx = np.absolute(
                z10_file_matrix[:, 0] - Lon) + np.absolute(z10_file_matrix[:, 1] - Lat)
            index = np.argmin(dist_approx)
            z10_dict[self.nodes[i]] = z10_file_matrix[index, 2]
        save_obj(z10_dict, 'z10_dict')

    def visualize_adjacency_matrix(self, method="byRandom", figure_file_name="test.png"):
        A = self.get_adjacency_matrix()
        self.visualize_matrix(A, figure_file_name, method)

    def annotate_matrix_figure(self, fig, ax, node_community_size, rearranged_community_order=None):
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Node number')
        ax.set_ylabel('Node number')
        for i in range(len(node_community_size)):
            if node_community_size[i] > 11:
                ax.annotate('', xy=(-55, sum(node_community_size[:i]) - 3), xytext=(-55, sum(node_community_size[:(i+1)]) + 3), textcoords='data',
                            arrowprops=dict(arrowstyle='<->', facecolor='red'),
                            annotation_clip=False)
                if rearranged_community_order is None:
                    ax.annotate(str(i + 1), xy=(-70, (sum(node_community_size[:i]) + sum(node_community_size[:(i+1)])) / 2),
                                annotation_clip=False, bbox={"boxstyle": "circle", "fc": "w"}, size=20.0 / (i//9 + 2))
                else:
                    ax.annotate(str(rearranged_community_order[i] + 1), xy=(-70, (sum(node_community_size[:i]) + sum(node_community_size[:(i+1)])) / 2),
                                annotation_clip=False, bbox={"boxstyle": "circle", "fc": "w"}, size=20.0 / (rearranged_community_order[i]//9 + 2))

    def visualize_matrix(self, A, figure_file_name, method="byRandom"):
        fig, ax = plt.subplots()
        if method == "byRandom":
            index = np.random.choice(
                np.arange(len(self.nodes)), size=len(self.nodes), replace=False)
        if method == "byCommunity":
            node_community = [[j for j, lst in enumerate(
                self.partition) if node in lst][0] for node in self.nodes]
            index = np.argsort(node_community)
        if method == "original":
            index = np.arange(len(self.nodes))
        if method == "byCommunitySize":
            node_community = [[j for j, lst in enumerate(
                self.partition) if node in lst][0] for node in self.nodes]
            node_community_size = np.array(
                [len(self.partition[community]) for community in node_community])
            index = np.lexsort((node_community, -node_community_size))
            node_community_size_unique = np.sort(
                [len(k) for k in self.partition])[::-1]
            self.annotate_matrix_figure(fig, ax, node_community_size_unique)
        if method == "byCommunitySize2":
            node_community = [[j for j, lst in enumerate(
                self.partition) if node in lst][0] for node in self.nodes]
            node_community_size = np.array(
                [len(self.partition[community]) for community in node_community])
            indexnew = np.lexsort((node_community, -node_community_size))
            node_community_size_unique = np.sort(
                [len(k) for k in self.partition])[::-1]
            rearranged_community_order = np.array(
                [0, 1, 5, 2, 4, 7, 3, 9, 6, 8, 10, 11, 12, 13, 14])
            index = []
            for i in rearranged_community_order:
                index.append(indexnew[sum(node_community_size_unique[:i]):sum(
                    node_community_size_unique[:(i+1)])])
            node_community_size_unique_rearranged = node_community_size_unique[
                rearranged_community_order]
            self.annotate_matrix_figure(
                fig, ax, node_community_size_unique_rearranged, rearranged_community_order)
            index = np.concatenate(index).ravel()
        cmap = plt.cm.seismic
        norm = matplotlib.colors.Normalize(vmin=-6, vmax=6)
        im = ax.imshow(A[:, index][index], cmap=cmap, norm=norm)
        #cax = fig.add_axes([0.9, 0.2, 0.01, 0.5])
        #fig.colorbar(im, cax=cax)
        fig.savefig(figure_file_name, bbox_inches='tight')
        plt.close(fig)

    def plot_eigen_gaps(self, number_of_eigenval, figure_file_name):
        eigen_gaps = compute_eigen_gaps(
            self.get_adjacency_matrix(), number_of_eigenval)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(np.arange(2, number_of_eigenval),
                eigen_gaps[1:(number_of_eigenval - 1)], 'o')
        ax.set_ylabel('$\delta_k$')
        ax.set_xlabel('$k$')
        ax.set_xticks(np.arange(2, number_of_eigenval))
        fig.tight_layout()
        fig.savefig(figure_file_name, dpi=300)

    def buildEmpiricalModel(method, obj='dataDictUnsmoothed', minNumberForCorrelationCoefficient=7, maxDistanceForCorrelationCoefficient=100, binsize=10, R=100):
        exists = os.path.isfile('obj/' + obj + '.pkl')
        if exists:
            dataDict = load_obj(obj)
        else:
            dataDict = saveDataDict()
        siteID = dataDict['siteID']
        uniqueSiteID = np.sort(np.unique(siteID))
        dist = []
        coff = []
        for i in range(len(uniqueSiteID)):
            for j in range(i + 1, len(uniqueSiteID)):
                correlationCoefficient = getCorrelationCoefficient(
                    uniqueSiteID[i], uniqueSiteID[j], dataDict, minNumberForCorrelationCoefficient)
                distance = getDistance(
                    uniqueSiteID[i], uniqueSiteID[j], dataDict)
                if correlationCoefficient != None and distance < maxDistanceForCorrelationCoefficient:
                    dist.append(distance)
                    coff.append(correlationCoefficient)
        avergaedDist, averagedCoff = meanInEachBin(dist, coff, binsize, R)
        empiricalModel = empiricalSpatialCorrelationModel(
            method, data=[avergaedDist, averagedCoff])
        return empiricalModel

    def visualize_correlation_distance(self, min_number_for_rho=7, max_distance_for_rho=100, figure_file_name="test.png", binsize=20, R=101):
        site_id = self.data_dict['siteID']
        unique_site_id = np.unique(site_id)
        dist = []
        coff = []
        for i in range(len(unique_site_id)):
            for j in range(i + 1, len(unique_site_id)):
                rho = get_correlation_coefficient(
                    unique_site_id[i], unique_site_id[j], self.data_dict, min_number_for_rho)
                distance = get_distance(
                    unique_site_id[i], unique_site_id[j], self.data_dict)
                if rho != None and distance < max_distance_for_rho:
                    dist.append(distance)
                    coff.append(rho)
        #dist_average, coff_average = mean_in_each_bin(dist, coff, binsize, R)
        empirical_correlation_model = EmpiricalSpatialCorrelationModel(
            "weighted_least_square", data=[dist, coff], search_range = (79, 81))
        self.empirical_correlation_model_global = empirical_correlation_model
        fig = plt.figure(figsize=(8, 8))
        plt.plot(dist, coff, '.', markersize=1,
                 label='Correlation Coefficient')
        plt.plot(np.arange(0, R, 10), empirical_correlation_model.predict(1, np.arange(
            0, R, 10)), '-k', label='Fitted Global Correlation Function', linewidth=3)
        #plt.plot(dist_average, coff_average, '*k', markersize = 8, label = 'Averaged Correlation Coefficient')
        plt.xlabel('Distance [km]', fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.ylabel('Correlation Coefficient', fontsize=15)
        plt.legend(prop={'size': 15}, loc='upper right')
        plt.ylim([-0.5, 1])
        fig.savefig(figure_file_name)
        fig.clf()
        print(empirical_correlation_model.range)
        print(empirical_correlation_model.predict(1, empirical_correlation_model.range))

    def correlation_distance_plot(self, dist, coff, figure_file_name, *args, binsize=20, R=101, **kwargs):
        dist_average, coff_average = mean_in_each_bin(dist, coff, binsize, R)
        empirical_correlation_model = EmpiricalSpatialCorrelationModel(
            "fitted_exponential", data=[dist_average, coff_average]).predict
        fig = plt.figure(figsize=(8, 8))
        plt.plot(dist, coff, '.', markersize=1,
                 label='Correlation Coefficient')
        plt.plot(np.arange(0, R, 10), empirical_correlation_model(1, np.arange(
            0, R, 10)), *args, **kwargs)
        plt.xlabel('Distance [km]', fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.ylabel('Correlation Coefficient', fontsize=15)
        plt.legend(prop={'size': 15}, loc='upper right')
        plt.ylim([-0.5, 1])
        fig.savefig(figure_file_name)
        fig.clf()

    def get_correlation_coefficient_within_community(self, community, min_number_for_rho, max_distance_for_rho):
        dist_community = []
        coff_community = []
        for i in range(len(community)):
            for j in range(i + 1, len(community)):
                rho = get_correlation_coefficient(
                    self.nodes[community[i]], self.nodes[community[j]], self.data_dict, min_number_for_rho)
                distance = get_distance(
                    self.nodes[community[i]], self.nodes[community[j]], self.data_dict)
                if rho != None and distance < max_distance_for_rho:
                    dist_community.append(distance)
                    coff_community.append(rho)
        return dist_community, coff_community

    def get_correlation_coefficient_across_community(self, community1, community2, min_number_for_rho, max_distance_for_rho):
        dist_community = []
        coff_community = []
        for i in range(len(community1)):
            for j in range(len(community2)):
                rho = get_correlation_coefficient(
                    self.nodes[community1[i]], self.nodes[community2[j]], self.data_dict, min_number_for_rho)
                distance = get_distance(
                    self.nodes[community1[i]], self.nodes[community2[j]], self.data_dict)
                if rho != None and distance < max_distance_for_rho:
                    dist_community.append(distance)
                    coff_community.append(rho)
        return dist_community, coff_community

    def visualize_correlation_distance_by_community(self, min_number_for_rho=7, max_distance_for_rho=100, figure_file_name="test.png", binsize=20, R=101):
        cmap = plt.cm.seismic
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(self.partition))
        community_name = ['Community 1', 'Communities 2 & 3', 'Community 4']
        community_symbol = ['o', 's', '^']
        community_color = [1, 3, 4]
        search_range_list = [(224, 226), (167, 169), (55, 57)]
        empirical_correlation_models = []
        community_copy = self.partition.copy()
        community_merged = []
        community_merged.append(community_copy[1])
        community_merged.append(np.concatenate((community_copy[0], community_copy[2], community_copy[3])))
        community_merged.append(community_copy[4])
        for community_index, community in enumerate(community_merged):
            dist_community, coff_community = self.get_correlation_coefficient_within_community(community, min_number_for_rho, max_distance_for_rho)
            self.correlation_distance_plot(dist_community, coff_community, 'figure/coff_dist_PH_community_' + community_name[community_index] + '.png', '-*', color=cmap(norm(community_color[community_index])), label=community_name[community_index], marker = community_symbol[community_index], linewidth=2)
            #dist_average, coff_average = mean_in_each_bin(dist_community, coff_community, binsize, R)
            empirical_correlation_model_community = EmpiricalSpatialCorrelationModel(
                "weighted_least_square", data=[dist_community, coff_community], search_range = search_range_list[community_index])
            empirical_correlation_models.append(empirical_correlation_model_community)

        # for i in range(len(community_merged)):
        #     for j in range(i + 1, len(community_merged)):
        #         dist_community, coff_community = self.get_correlation_coefficient_across_community(community_merged[i], community_merged[j], min_number_for_rho, max_distance_for_rho)
        #         self.correlation_distance_plot(dist_community, coff_community, 'figure/coff_dist_PH_community_' + community_name[i] + '-' + community_name[j] + '.png', '--k', label=community_name[i] + '-' + community_name[j], linewidth=2)
        #

        dist_across_community, coff_across_community = [], []
        for i in range(len(community_merged)):
            for j in range(i + 1, len(community_merged)):
                dist_community, coff_community = self.get_correlation_coefficient_across_community(community_merged[i], community_merged[j], min_number_for_rho, max_distance_for_rho)
                dist_across_community.extend(dist_community)
                coff_across_community.extend(coff_community)
        #self.correlation_distance_plot(dist_across_community, coff_across_community, 'figure/coff_dist_PH_across_community.png', '--k', label='Across Communities Correlation', linewidth=2)
        #dist_average, coff_average = mean_in_each_bin(dist_across_community, coff_across_community, binsize, R)
        empirical_correlation_model_across_community = EmpiricalSpatialCorrelationModel(
                "weighted_least_square", data=[dist_across_community, coff_across_community], search_range = (23, 25))

        fig = plt.figure(figsize=(8, 8))
        for i, empirical_correlation_model in enumerate(empirical_correlation_models):
            plt.plot(np.arange(0, R, 10), empirical_correlation_model.predict(1, np.arange(
                0, R, 10)), '-*', color=cmap(norm(community_color[i])), label= community_name[i] + ', $\hat{r}$ = ' + "{:.1f}".format(empirical_correlation_model.range), marker = community_symbol[i], linewidth=2)
        plt.plot(np.arange(0, R, 1), self.empirical_correlation_model_global.predict(1, np.arange(
            0, R, 1)), '-k', label='Global' + ', $\hat{r}$ = ' + "{:.1f}".format(self.empirical_correlation_model_global.range), linewidth=2)
        plt.plot(np.arange(0, R, 1), empirical_correlation_model_across_community.predict(1, np.arange(
            0, R, 1)), '--k', label='Across Communities' + ', $\hat{r}$ = ' + "{:.1f}".format(empirical_correlation_model_across_community.range), linewidth=2)
        plt.xlabel('Distance [km]', fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.ylabel('Correlation Coefficient', fontsize=15)
        # handles, labels = plt.gca().get_legend_handles_labels()
        # order = [2,1,3,4,0]
        plt.legend(prop={'size': 15}, loc='upper right')
        fig.savefig(figure_file_name)
