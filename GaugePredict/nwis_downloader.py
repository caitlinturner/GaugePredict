import pandas as pd
from dataretrieval import nwis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def get_sites_by_huc(huc_codes, parameter_code):
    site_info_list = []
    for huc_code in huc_codes:
        sites, metadata = nwis.what_sites(huc=huc_code, parameterCd=parameter_code)
        site_info_list.append(sites)
    site_info_df = pd.concat(site_info_list, ignore_index=True)
    return site_info_df

def create_site_info_df(sites, batch_size=100):
    site_info_list = []
    for i in range(0, len(sites), batch_size):
        site_info, metadata = nwis.get_info(sites=sites[i:i+batch_size])
        site_info_list.append(site_info)
    site_info_df = pd.concat(site_info_list, ignore_index=True)
    return site_info_df

def cluster_sites(sites, cluster_columns, n_clusters, inplace=False, random_state=None):
    features = sites[cluster_columns]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    if inplace:
        sites = sites.copy()
    sites['cluster'] = kmeans.fit_predict(features_scaled)
    return sites

def get_site_subsamples(clustered_sites):
    smallest_cluster_size = clustered_sites['cluster'].value_counts().min()
    subsamples = [[] for _ in range(smallest_cluster_size)]
    for cluster in clustered_sites['cluster'].unique():
        cluster_sites = clustered_sites[clustered_sites['cluster'] == cluster]
        for i, subsample in enumerate(subsamples):
                subsample.append(cluster_sites['site_no'].iloc[i])
    return subsamples
