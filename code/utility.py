import numpy as np
import geopy.distance
import pickle
from scipy.linalg import eigh


def mean_in_each_bin(dist, coff, binsize=10, R=100):
    dist = np.array(dist)
    coff = np.array(coff)
    edges = [0]
    for i in range(1, binsize + 1):
        edges.append(R / binsize * i)
    x = []
    y = []
    for i in range(binsize):
        x.append((edges[i] + edges[i+1]) / 2)
        index = np.logical_and(dist > edges[i], dist < edges[i + 1])
        y.append(np.nanmean(coff[index]))
    return x, y


def is_in_region_box(location, lon1=174.5, lon2=175.5, lat1=-41.5, lat2=-41):
    return(location[0] > lon1 and location[0] < lon2 and location[1] > lat1 and location[1] < lat2)


def fisher_transform(x):
    return(0.5 * np.log((1.0+x)/(1.0-x)))


def inverse_fisher_transform(x):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


def transform_e_to_rho(e, rho, n):
    return inverse_fisher_transform(e / np.sqrt(n - 3) + fisher_transform(rho))


def get_number_of_common_eq(site_id_1, site_id_2, data_dict):
    eq_id = data_dict['eqID']
    site_id = data_dict['siteID']
    eq_id_site1 = eq_id[site_id == site_id_1]
    eq_id_site2 = eq_id[site_id == site_id_2]
    number_of_common_eq = len(np.intersect1d(eq_id_site1, eq_id_site2))
    return number_of_common_eq


def get_correlation_coefficient(site_id1, site_id2, data_dict, min_number_for_rho=7):
    eqID = data_dict['eqID']
    site_id = data_dict['siteID']
    residual = data_dict['residual']
    residual_site1 = residual[site_id == site_id1]
    eq_id_site1 = eqID[site_id == site_id1]
    residual_site2 = residual[site_id == site_id2]
    eq_id_site2 = eqID[site_id == site_id2]
    common_eq_id = np.intersect1d(eq_id_site1, eq_id_site2)
    residual_sorted_by_eqid_site1 = np.array(
        [residual_site1[eq_id_site1 == x] for x in common_eq_id])
    residual_sorted_by_eqid_site2 = np.array(
        [residual_site2[eq_id_site2 == x] for x in common_eq_id])
    if len(common_eq_id) > min_number_for_rho:
        result = np.corrcoef(residual_sorted_by_eqid_site1.T,
                             residual_sorted_by_eqid_site2.T)[0, 1]
        # if one of the residual is constant, standard deviation is zero
        if np.isnan(result):
            return None
        else:
            return result
    else:
        return None


def get_distance(site_id_1, site_id_2, data_dict):
    lat1 = data_dict['StationLoc'][np.where(
        data_dict['siteID'] == site_id_1)[0][0]][1]
    lon1 = data_dict['StationLoc'][np.where(
        data_dict['siteID'] == site_id_1)[0][0]][0]
    lat2 = data_dict['StationLoc'][np.where(
        data_dict['siteID'] == site_id_2)[0][0]][1]
    lon2 = data_dict['StationLoc'][np.where(
        data_dict['siteID'] == site_id_2)[0][0]][0]
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return(geopy.distance.vincenty(coords_1, coords_2).km)


def jayaram_baker_spatial_correlation(T, h, is_vs30_clustered=0):
    if T < 1:
        if is_vs30_clustered == 0:
            b = 8.5 + 17.2 * T
        elif is_vs30_clustered == 1:
            b = 40.7 - 15.0 * T
    elif T >= 1:
        b = 22.0 + 3.7 * T
    rho = np.exp(-3 * h / b)
    return rho


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    if name == None:
        return None
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')


def compute_eigen_gaps(A, number_of_eigenval):
    D_bar = np.diag(np.sum(np.absolute(A), axis=1).reshape(-1))
    L_bar = D_bar - A
    w, v = eigh(L_bar, eigvals=(0, number_of_eigenval))
    return np.diff(w)
