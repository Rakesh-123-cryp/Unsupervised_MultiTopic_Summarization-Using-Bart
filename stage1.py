from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.spatial.distance import cdist

#Get Embedding
def embedding():
    return

def elbow(embed, means):
    score = cdist(embed, means)
    score = np.sum(np.min(score, axis=1))/embed.shape[0]
    return score

def silhouette(embed, classes):
    scores = silhouette_score(embed, classes, metric="euclidean")
    return scores

#GMM clustering
def clustering(sentence_embeddings):
    cluster_count = 2
    scores = []
    labels_array = []
    while(cluster_count<5):
        gmm = GaussianMixture(cluster_count)
        labels = gmm.fit_predict(sentence_embeddings) # sentence_embedding shape -> (num_samples, num_features)
        labels_array.append(labels)
        scores.append(silhouette(sentence_embeddings, labels))
        #scores.append(elbow(sentence_embeddings, gmm.means_))
    
    least = np.argmin(scores)
    
    return labels[least]