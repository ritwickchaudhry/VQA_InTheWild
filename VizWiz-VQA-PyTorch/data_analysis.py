import os
import h5py as h5
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def pool_and_linearize(features):
    # Input - Features - N x C x H x W
    N, C, H, W = features.shape
    features = features.reshape((N, C, H * W)).mean(axis=2)
    return features

def load_and_prepare_features(path, N):
    h5File = h5.File(path, 'r')
    return pool_and_linearize(h5File['att'][:N])

def plot(ax, points, color, label, marker='.'):
    ax.scatter(points[:,0], points[:,1], c=color, label=label, marker=marker)

def plot_unperturbed():
    N = 500
    # VizWiz
    vizwiz_path = '/home/ubuntu/prepro_data/vizwiz/image_featurestrain.h5'
    vizwiz_features = load_and_prepare_features(vizwiz_path, N)
    
    # VQA v2.0
    vqa_path = '/home/ubuntu/train_feat.h5'
    vqa_features = load_and_prepare_features(vqa_path, N)
        
    all_features = np.concatenate([vizwiz_features, vqa_features], axis=0)
    tsne_feats = TSNE(n_components=2).fit_transform(all_features)
    labels = np.concatenate([np.zeros(N), np.ones(N)], axis=0)
    
    fig, ax = plt.subplots()
    vizwiz_feats_reduced = tsne_feats[:N, :]
    vqa_feats_reduced = tsne_feats[N:, :]
    ax.scatter(vqa_feats_reduced[:, 0], vqa_feats_reduced[:, 1], c=['blue' for i in range(vqa_feats_reduced.shape[0])], label='VizWiz', marker='.')
    ax.scatter(vizwiz_feats_reduced[:, 0], vizwiz_feats_reduced[:, 1], c=['red' for i in range(vizwiz_feats_reduced.shape[0])], label='VQA v2.0', marker='.')
    ax.legend()
    plt.title("ResNet-152 t-SNE")
    plt.savefig("TSNE.png")

def plot_perturbed():
    N = 100
    # VizWiz
    vizwiz_path = '/home/ubuntu/prepro_data/vizwiz/image_featurestrain.h5'
    vizwiz_features = load_and_prepare_features(vizwiz_path, N)
    
    # VQA v2.0
    vqa_no_perturb_path = '/home/ubuntu/prepro_data/vqa/no_perturb_feat.h5'
    vqa_low_perturb_path = '/home/ubuntu/prepro_data/vqa/low_perturb_feat.h5'
    vqa_mid_perturb_path = '/home/ubuntu/prepro_data/vqa/mid_perturb_feat.h5'
    vqa_no_perturb_features = load_and_prepare_features(vqa_no_perturb_path, N)
    vqa_low_perturb_features = load_and_prepare_features(vqa_low_perturb_path, N)
    vqa_mid_perturb_features = load_and_prepare_features(vqa_mid_perturb_path, N)


    all_features = np.concatenate([vizwiz_features, vqa_no_perturb_features, vqa_low_perturb_features, vqa_mid_perturb_features], axis=0)
    tsne_feats = TSNE(n_components=2).fit_transform(all_features)
    labels = np.concatenate([np.zeros(N), np.ones(N)], axis=0)
    
    fig, ax = plt.subplots()
    vizwiz_feats_reduced = tsne_feats[:N, :]
    
    vqa_no_perturb_feats_reduced = tsne_feats[N : 2 * N, :]
    vqa_low_perturb_feats_reduced = tsne_feats[2 * N : 3 * N, :]
    vqa_mid_perturb_feats_reduced = tsne_feats[3 * N : 4 * N, :]

    plot(ax, vizwiz_feats_reduced, 'blue', 'VizWiz', '.')
    plot(ax, vqa_no_perturb_feats_reduced, 'red', 'VQA 2.0', '+')
    plot(ax, vqa_low_perturb_feats_reduced, 'black', 'VQA 2.0 Low', '+')
    plot(ax, vqa_mid_perturb_feats_reduced, 'green', 'VQA 2.0 Mid', '+')

    # ax.scatter(
    #     vqa_feats_reduced[:, 0], 
    #     vqa_feats_reduced[:, 1], 
    #     c=['blue' for i in range(vqa_feats_reduced.shape[0])], 
    #     label='VizWiz', 
    #     marker='.'
    # )
    # ax.scatter(vizwiz_feats_reduced[:, 0], vizwiz_feats_reduced[:, 1], c=['red' for i in range(vizwiz_feats_reduced.shape[0])], label='VQA v2.0', marker='.')
    ax.legend()
    plt.title("ResNet-152 t-SNE")
    plt.savefig("TSNE.png")

if __name__ == "__main__":
    # plot_unperturbed()
    plot_perturbed()