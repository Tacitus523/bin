#!/home/lpetersen/anaconda_interpreter/envs/mda/bin/python3
import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import time

FONTSIZE = 30
LABELSIZE = 18
MARKERSIZE = 12
DPI=600

DEFAULT_DATA_SOURCE = "Unknown source"
ORIGINAL_DATA_SOURCE = ["Metadynamics","PES Scan"]

def main():
    ap = argparse.ArgumentParser(description="t-SNE analysis for .xyz-trajectories")
    ap.add_argument("-f", type=str, dest="file", action="store", required=True, help="File with geometry data to plot", metavar="file")
    ap.add_argument("-s", type=str, dest="data_source_file", action="store", required=False, default=None, help="File with data sources for coloring", metavar="data source file")
    args = ap.parse_args()
    file = args.file
    data_source_file = args.data_source_file

    u = mda.Universe(file)
    atoms = u.select_atoms('all')
    n_atoms = len(atoms)
    n_time_steps = len(u.trajectory)
    self_distances = distances.self_distance_array(atoms.positions)


    all_self_distances = np.zeros((n_time_steps, self_distances.shape[0]))
    all_sq_distances = np.zeros((n_time_steps, n_atoms, n_atoms))
    for time_step in u.trajectory:
        self_distances = distances.self_distance_array(atoms.positions)
        sq_distances = np.zeros((n_atoms, n_atoms))
        triu = np.triu_indices_from(sq_distances, k=1)
        sq_distances[triu] = self_distances
        sq_distances.T[triu] = self_distances

        all_self_distances[time_step.frame] = self_distances
        all_sq_distances[time_step.frame] = sq_distances

    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(all_self_distances)

    df = pd.DataFrame()
    df["PCA 1"] = pca_result[:,0]
    df["PCA 2"] = pca_result[:,1] 
    df["PCA 3"] = pca_result[:,2]
    if data_source_file is not None:
        df["data_source"] = pd.read_csv(data_source_file)
        df["original_data"] = df["data_source"]
        condition = df["data_source"].isin(ORIGINAL_DATA_SOURCE)
        # Assign a new constant value only when the condition is false
        df.loc[~condition, 'original_data'] = 'Adaptive Sampling'
    else:
        df["data_source"] = ["Unknown source"]*len(df)
        df["original_data"] = [True]*len(df)
        #print("Reading some hardcoded values for the data sources used for coloring. To prevent this, prepare a data source file and pass it with -s")
    # df["data_source"] = ["Energy Scan"]*5112 + ["Metadynamic"]*(13548-5112) + ["Adaptive Sampling 0"]*1000 + ["Adaptive Sampling 1"]*1000 + ["Adaptive Sampling 2"]*1000 + ["Adaptive Sampling 3"]*1000 + ["Adaptive Sampling 4"]*1000+ ["Adaptive Sampling 4"]*981
    # df["data_source"] = ["energy scan"]*5112 + ["metadynamic"]*(13548-5112) + ["random_displacement"]*9000 + ["energy scan"]*5112 + ["metadynamic"]*(13548-5112) + ["random_displacement"]*9000 + ["metadynamic protein env"]*9998
    df["data_source_codes"] = pd.Categorical(df["data_source"]).codes

    print(f"Explained variation per first 3 principal component: {pca.explained_variance_ratio_[:3]}")
    print(f"Explained variation sum of principal components: {np.sum(pca.explained_variance_ratio_)}")

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()
    sns.scatterplot(
    x="PCA 1", y="PCA 2",
    hue="data_source",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3)
    ax.set_xlabel('PCA 1', fontsize=FONTSIZE)
    ax.set_ylabel('PCA 2', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)
    legend = plt.legend(title='Data Source', loc='upper left', fontsize=FONTSIZE, title_fontsize=FONTSIZE, bbox_to_anchor=(1.05, 1))
    for legend_handle in legend.legendHandles: 
        legend_handle.set_alpha(1)
        legend_handle.set_markersize(MARKERSIZE)
    plt.tight_layout()
    plt.savefig("2D_PCA_plot.png", dpi=DPI)
    plt.close()
    
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(
        xs=df["PCA 1"], 
        ys=df["PCA 2"], 
        zs=df["PCA 3"], 
        c=df["data_source_codes"], 
        cmap='tab10'
    )
    ax.set_xlabel('PCA 1', fontsize=FONTSIZE)
    ax.set_ylabel('PCA 2', fontsize=FONTSIZE)
    ax.set_zlabel('PCA 3', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)
    legend = ax.legend(*[scatter.legend_elements()[0], 
                    ["energy_scan", "metadynamic"]], 
                    #["energy_scan", "metadynamic", "metadynamic protein env", "random_displacement"]], 
                    title='Data Source', loc='upper left', fontsize=FONTSIZE, title_fontsize=FONTSIZE)
    for legend_handle in legend.legendHandles: 
        legend_handle.set_alpha(1)
        legend_handle.set_markersize(MARKERSIZE)
    ax.add_artist(legend)
    plt.tight_layout()
    plt.savefig("3D_PCA_plot.png", dpi=DPI)
    plt.close()

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
    tsne_results = tsne.fit_transform(all_self_distances)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df['tSNE 1'] = tsne_results[:,0]
    df['tSNE 2'] = tsne_results[:,1]

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()
    pallete = sns.color_palette("dark:#5A9", df["data_source"].nunique()) # Pallete with just enough colors
    markers = ["s","^","o"]
    for i in range(len(ORIGINAL_DATA_SOURCE)):
        pallete[i] = pallete[0] 
    sns.scatterplot(
        data=df,
        x="tSNE 1", y="tSNE 2",
        hue="data_source",
        palette=pallete,
        style="original_data",
        style_order=ORIGINAL_DATA_SOURCE+['Adaptive Sampling'],
        markers=markers,
        legend="full",
        alpha=0.5
    )
    
    ax.set_xlabel('Arbitrary tSNE axis 1',fontsize=FONTSIZE)
    ax.set_ylabel('Arbitrary tSNE axis 2',fontsize=FONTSIZE)
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)
    legend = plt.legend(title='Data Source', loc='upper left', fontsize=FONTSIZE, title_fontsize=FONTSIZE, bbox_to_anchor=(1.05, 1))
    
    # for legend_handle in legend.legendHandles: 
    #     legend_handle.set_alpha(1)
    #     legend_handle.set_markersize(MARKERSIZE)

    # Create a custom legend with modified handles and labels
    legend_handles, legend_labels = ax.get_legend_handles_labels()

    visible_indices = list(range(1,len(ORIGINAL_DATA_SOURCE)+2)) + [df["data_source"].nunique()-1, df["data_source"].nunique()]
    visible_legend_handles = [legend_handles[i] for i in visible_indices]

    for i, legend_handle in enumerate(visible_legend_handles):
        legend_handle.set_color(pallete[visible_indices[i]-1])
        legend_handle.set_alpha(1)
        if i<len(ORIGINAL_DATA_SOURCE):
            legend_handle.set_marker(markers[i])
        else:
            legend_handle.set_marker(markers[-1])
        legend_handle.set_markersize(MARKERSIZE)
    visible_legend_handles[-2].set_visible(False)

    visible_legend_labels = [legend_labels[i] for i in visible_indices]
    visible_legend_labels[-2] = "..."

    ax.legend(handles=visible_legend_handles, labels=visible_legend_labels)
    plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text

    plt.tight_layout()
    plt.savefig("2D_tSNE_plot_new.png", dpi=DPI, bbox_inches = "tight")
    plt.close()

main()