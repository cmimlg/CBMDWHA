
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from itertools import cycle
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import DBSCAN
import os
import logging
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
import matplotlib.cm as cm
from sklearn.decomposition import TruncatedSVD

import scipy.io
from skfeature.function.similarity_based import SPEC
from skfeature.utility import unsupervised_evaluation
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.cluster import adjusted_rand_score
from pyclustering.cluster.birch import birch
from sklearn.neighbors import NearestNeighbors
import operator

# create logger for the application
logger = logging.getLogger('CMDWABD Logger')

ch = logging.StreamHandler()

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)


logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

def read_data():
    os.chdir("/home/admin123/Clustering_MD/Paper/clustering.experiments/")
    fp = "Synthetic_Data_Recoded.csv"
    df = pd.read_csv(fp)
    gt_assgn = df["C"]
    cols_req = df.columns.tolist()
    df = df[cols_req]
    return df, gt_assgn

def do_mini_batch_km_range():
    df, gt_assgn = read_data()
    X = df.as_matrix()
    
    krange = np.arange(start = 2, stop = 6, step = 1)
    ari_values = []
    for k in krange:
        # Compute clustering with MiniBatchKMeans.
        print ("Calculating solution for k = " + str(k))
        mbk = MiniBatchKMeans(init='k-means++', n_clusters= k, batch_size= 2000,
                      n_init=10, max_no_improvement=10, verbose=0,
                      random_state=0)
        mbk.fit(X)
        clus_labels = mbk.labels_
        ari = adjusted_rand_score(gt_assgn, clus_labels)
        ari_values.append(ari)

    

    plt.scatter(krange, ari_values, color = "red")
    plt.title("ARI Versus K - Synthetic Dataset, batch-size = 2000")
    plt.xlim([0,10])
    plt.xlabel("K")
    plt.ylabel("ARI")
    plt.grid()
    plt.show()

    return ari_values

def do_fs_mini_batch_km_range():
    df, gt_assgn = read_data()
    X = df.as_matrix()
    ipca = IncrementalPCA(n_components = 2)
    X_ipca = ipca.fit_transform(X)
    del X
    krange = np.arange(start = 2, stop = 6, step = 1)
    ari_values = []
    for k in krange:
        # Compute clustering with MiniBatchKMeans.
        print ("Calculating solution for k = " + str(k))
        mbk = MiniBatchKMeans(init='k-means++', n_clusters= k, batch_size= 2000,
                      n_init=10, max_no_improvement=10, verbose=0,
                      random_state=0)
        mbk.fit(X_ipca)
        clus_labels = mbk.labels_
        ari = adjusted_rand_score(gt_assgn, clus_labels)
        ari_values.append(ari)

    

    plt.scatter(krange, ari_values, color = "red")
    plt.title("ARI Versus K With FE- Synthetic Dataset, batch-size = 2000")
    plt.xlim([0,10])
    plt.xlabel("K")
    plt.ylabel("ARI")
    plt.grid()
    plt.show()

    return

def do_mini_batch_kmeans():
    fp = "/home/admin123/Clustering_MD/Paper/clustering.experiments/"\
    "Jan_2016_Delays_Recoded.csv"
    df = pd.read_csv(fp)
    X = df.as_matrix()
    mbk = MiniBatchKMeans(init='k-means++', n_clusters= 35, batch_size=100,
                      n_init=10, max_no_improvement=10, verbose=0,
                      random_state=0)

    
    t0 = time()
    mbk.fit(X)
    t_mini_batch = time() - t0
    print("Time taken to run MiniBatchKMeans %0.2f seconds" % t_mini_batch)
    mbk_means_labels_unique = np.unique(mbk.labels_)
    df.loc[:,"Cluster"] = mbk.labels_
    fp_out = "/home/admin123/Clustering_MD/Paper/clustering.experiments/" \
            "jan2016_delay_data_clustered.csv"
    df.to_csv(fp_out, index = False)
    

    print("Done with Minibatch K-Means, starting incremental PCA...")
    ipca = IncrementalPCA(n_components = 2)
    X_ipca = ipca.fit_transform(X)
    
    # Use all colors that matplotlib provides by default.
    colors_ = cycle(colors.cnames.keys())

    ax = plt.gca()
    n_clusters = 35
    for this_centroid, k, col in zip(mbk.cluster_centers_,
                                     range(n_clusters), colors_):
        mask = mbk.labels_ == k
        ax.plot(X_ipca[mask, 0], X_ipca[mask, 1], 'w', markerfacecolor=col, marker='.')


    ax.set_title("Mini Batch KMeans Airline Delay for January 2016")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.grid()

    plt.show()

    return






def do_inc_pca():
    fp = "/home/admin123/Big_Data_Paper_Code_Data/HD_problems/CaliforniaHousing/cal_housing.csv"
    df = pd.read_csv(fp)
    X = df.as_matrix()

    ipca = IncrementalPCA(n_components = 2, batch_size = 100, whiten = True)
    X_ipca = ipca.fit_transform(X)




    return ipca, df, X_ipca


def do_DBSCAN():
    df, gt_assgn = read_data()
    X = df.as_matrix()
    del df
    logger.debug("Starting DBSCAN on large dataset - " + str(X.shape[0]) + " rows!")
    db = DBSCAN(eps= 0.4, min_samples = 10)
    db = db.fit(X)
    logger.debug("Done with DBSCAN !")
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    ari = adjusted_rand_score(gt_assgn, labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.debug("ARI is  " + str(ari))
    logger.debug("Number of Clusters is : " + str(n_clusters))

    return

def do_GS_DBSCAN():
    df, gt_assgn = get_sample()
    X = df.as_matrix()
    del df
    eps_range = np.linspace(0.01, 0.5, 10)
    ns_range = np.linspace(5, 1000, 20, dtype = np.int)
    res_dict = dict()
    

    for e, n in [(e, n) for e in eps_range for n in ns_range]:
        logger.debug("running eps = " + str(e) + " n = " + str(n))
        
        db = DBSCAN(eps= e, min_samples = n)
        db = db.fit(X)
        logger.debug("Done with DBSCAN !")
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        ari = adjusted_rand_score(gt_assgn, labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        key = "Eps = " + str(e) + " n = " + str(n)
        res_dict[key] = ari


    return res_dict

def do_BIRCH(nc = 100):
    os.chdir("/home/admin123/Clustering_MD/Paper/clustering.experiments/")
    fp = "Jan_2016_Delays_Recoded.csv"
    df = pd.read_csv(fp)
    X = df.as_matrix()
    del df
    ipca = IncrementalPCA(n_components = 2)
    X_ipca = ipca.fit_transform(X)
    del X

    
    logger.debug("Starting BIRCH on large dataset - " + str(X_ipca.shape[0]) + " rows!")
    brc = Birch(branching_factor=50, n_clusters=nc,\
                threshold=0.25,compute_labels=True)
    brc = brc.fit(X_ipca)
    labels = brc.predict(X_ipca)
    logger.debug("Done with BIRCH !")    
    chis = metrics.calinski_harabaz_score(X_ipca, labels)
    logger.debug("CH index score : " + str(chis))
    colors = cm.rainbow(np.linspace(0, 1, nc))
    ax = plt.gca()
    for l,c  in zip(labels, colors):
        mask = labels == l
        ax.plot(X_ipca[mask, 0], X_ipca[mask, 1], 'w',\
                    markerfacecolor=c , marker='.')
    ax.set_title("BIRCH Airline Delay for January 2016")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.grid()
    plt.show()
        

    return



def do_BIRCH_K_Range(low = 2, high = 6):
    df, gt_assgn = read_data()
    X = df.as_matrix()

    logger.debug("Starting BIRCH on large dataset - " +\
                 str(X.shape[0]) + " rows!")
    kvals = np.arange(start = low, stop = high, step = 1)
    ari_dict = dict()
    for k in kvals:
        logger.debug("Running k = " + str(k))
        brc = Birch(n_clusters=k, threshold= 1.0)

        brc = brc.fit(X)
        logger.debug("Done fitting !")
        clus_labels = brc.labels_
        logger.debug("Done with BIRCH !")    
        ari = adjusted_rand_score(gt_assgn, clus_labels)
        ari_dict[k] = ari
    logger.debug("Done with BIRCH!")
    ax = plt.gca()
    
    plt.scatter(ari_dict.keys(), ari_dict.values(), color = "red")
    ax.set_title("ARI versus K - BIRCH")
    ax.set_xlabel("K")
    ax.set_ylabel("ARI")
    plt.grid()
    plt.show()

    return ari_dict

def do_FS_DBSCAN():
    X_ipca = get_data()

    logger.debug("Starting DBSCAN on large dataset - " + str(X_ipca.shape[0]) + " rows!")
    db = DBSCAN(eps= 0.05, min_samples = 100, n_jobs = 2)
    db = db.fit(X_ipca)
    logger.debug("Done with DBSCAN !")
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    chis = metrics.calinski_harabaz_score(X_ipca, labels)
    logger.debug("CH index score : " + str(chis))
    logger.debug("Number of Clusters : " + str(n_clusters_))
    return 

def get_data():
    os.chdir("/home/admin123/Clustering_MD/Paper/clustering.experiments/")
    fp = "Jan_2016_Delays_Recoded.csv"
    df = pd.read_csv(fp)
    X = df.as_matrix()
    del df
    ipca = IncrementalPCA(n_components = 2, whiten = True)
    X_ipca = ipca.fit_transform(X)
    del X

    return X_ipca



def gen_data(write = True):
    os.chdir("/home/admin123/Clustering_MD/Paper/clustering.experiments/")
    fp = "Syn_Mixed_Data.csv"
    X, y = make_blobs(n_samples= int(1e6), centers=3,\
                      n_features=2, random_state = 0)
    df = pd.DataFrame(X)
    df["C"] = y
    df.columns = ["N1", "N2", "C"]
    
    df["C1"] = df.apply(set_c1vals, axis = 1)
    df["C2"] = df.apply(set_c2vals, axis = 1)

    if write:
        df.to_csv(fp, index = False)
    return df

def set_c1vals(row):
    c1_vals = ["CAT_VAR1_VAL_" + str(i+1) for i in range(3)]
    
    if row["C"] == 0:
        c1_mn = np.random.multinomial(1, (0.7, 0.15, 0.15), 1)
        c1_val = c1_vals[np.argmax(c1_mn)]
    elif row["C"] == 1:
        c1_mn = np.random.multinomial(1, (0.15, 0.7, 0.15), 1)
        c1_val = c1_vals[np.argmax(c1_mn)]
    else:
        c1_mn = np.random.multinomial(1, (0.15, 0.15, 0.7), 1)
        c1_val = c1_vals[np.argmax(c1_mn)]

    return c1_val

def set_c2vals(row):
    
    c2_vals = ["CAT_VAR2_VAL_" + str(i+1) for i in range(3)]
    if row["C"] == 0:
        c2_mn = np.random.multinomial(1, (0.8, 0.15, 0.05), 1)
        c2_val = c2_vals[np.argmax(c2_mn)]
    elif row["C"] == 1:
        c2_mn = np.random.multinomial(1, (0.05, 0.8, 0.15), 1)
        c2_val = c2_vals[np.argmax(c2_mn)]
    else:
        c2_mn = np.random.multinomial(1, (0.15, 0.05, 0.8), 1)
        c2_val = c2_vals[np.argmax(c2_mn)]

    return c2_val

def get_sample(frac = 0.01):

    os.chdir("/home/admin123/Clustering_MD/Paper/clustering.experiments/")
    fp = "Synthetic_Data_Recoded.csv"
    df = pd.read_csv(fp)
    df_sample = df.sample(frac = frac)
    df_sample_gtl = df_sample["C"]
    cols_req = df_sample.columns.tolist()
    cols_req.remove("C")
    df_sample = df_sample[cols_req]
    return df_sample, df_sample_gtl

def gen_knn_graph():
    df, lbl = get_sample()
    X = df.as_matrix()
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
    dist, _ = nbrs.kneighbors(X)
    kng = {i:dist[i,9] for i in range(X.shape[0])}
    sorted_kng = sorted(kng.items(), key=operator.itemgetter(1),\
                        reverse = True)
    vals = [v[1] for v in sorted_kng]
    inds = [ i for i in range(len(vals))]
    plt.scatter(inds, vals, color = "red")
    plt.grid()
    plt.show()
    return 
    
    
    

    

