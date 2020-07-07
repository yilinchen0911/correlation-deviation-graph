from SpatialGraph import SpatialGraph
import seaborn as sns
from EmpiricalSpatialCorrelationModel import EmpiricalSpatialCorrelationModel
sns.set()

if __name__ == '__main__':
    spatial_graph_cyberShake_PH = SpatialGraph(graph_path="graph/graphCyberShakePH_balanced.txt",
                                               data_dict_path="dataDictCyberShakePH", map_limit={'lat_0': 34, 'lon_0': -118.2, 'width': 2E5, 'height': 2E5})
    spatial_graph_cyberShake_PH.visualize_correlation_distance(
        figure_file_name="figure/coff_dist_PH.png")
    spatial_graph_cyberShake_PH.plot_eigen_gaps(
        16, figure_file_name="figure/eigen_gap_PH.png")
    spatial_graph_cyberShake_PH.clustering(
        method="Spectral", num_clust=5, num_eig=3)
    sns.reset_orig()
    spatial_graph_cyberShake_PH.visualize_adjacency_matrix(
        method="byCommunitySize", figure_file_name="figure/CyberShakePHSpectralAdjacencyMatrix_num_clust_5_balanced.png")
    spatial_graph_cyberShake_PH.visualize_community_by_z10(
        file_name="figure/community_by_z10_5.png")
    spatial_graph_cyberShake_PH.write_community_location_file(
        command_file='balanced5.sh', station_file_dir='balanced5', path="figure/GMT")
    spatial_graph_cyberShake_PH.clustering(
        method="Spectral", num_clust=15, num_eig=3)
    spatial_graph_cyberShake_PH.write_community_location_file(
        command_file='balanced15.sh', station_file_dir='balanced15', path="figure/GMT")
    spatial_graph_cyberShake_PH.visualize_adjacency_matrix(
        method="byCommunitySize2", figure_file_name="figure/CyberShakePHSpectralAdjacencyMatrix_num_clust_15_balanced.png")
    spatial_graph_cyberShake_PH.visualize_community_by_z10(
        file_name="figure/community_by_z10_15.png")
