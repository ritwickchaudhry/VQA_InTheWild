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

if __name__ == "__main__":
    N = 500
    # VizWiz
    vizwiz_path = '/home/ubuntu/prepro_data/image_featurestrain.h5'
    h5File = h5.File(vizwiz_path, 'r')
    vizwiz_features = pool_and_linearize(h5File['att'][:N])
    
    # VQA v2.0
    vqa_path = '/home/ubuntu/train_feat.h5'
    h5File = h5.File(vqa_path, 'r')
    vqa_features = pool_and_linearize(h5File['att'][:N])
        
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