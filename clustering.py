import random
import torch
import nltk
from sentence_transformers import SentenceTransformer

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.spatial.distance import cdist

nltk.download('punkt')
sent_embedding_model = "all-MiniLM-L6-v2"
device_used = "cuda"
random_seed = 70
random.seed(random_seed)

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

def split_sentence(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

def sent_embedding(paragraph):
    sentences = split_sentence(paragraph)
    if torch.cuda.is_available():
        model = SentenceTransformer(sent_embedding_model, device=device_used)
    else:
        model = SentenceTransformer(sent_embedding_model)
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings
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
    prob_array = []
    while(cluster_count<5):
        gmm = GaussianMixture(cluster_count)
        labels = gmm.fit_predict(sentence_embeddings) # sentence_embedding shape -> (num_samples, num_features)
        prob_array.append(gmm.predict_proba(sentence_embeddings))
        labels_array.append(labels)
        scores.append(silhouette(sentence_embeddings, labels))
        # print(scores)
        #scores.append(elbow(sentence_embeddings, gmm.means_))
        cluster_count+=1

    least = np.argmax(scores)

    return labels_array[least], prob_array


#Test sentence Embeddings

# embeddings = sent_embedding("Machine learning is part of data science. A movie theater, cinema, or cinema hall, also known as a movie house, picture house, picture theater or simply theater. Data science is part of the curriculum in many courses. The film is projected with a movie projector onto a large projection screen at the front of the auditorium while the dialogue, sounds and music are played through speakers. Movie theatres stand in a long tradition of theaters that could house all kinds of entertainment. we are doing NLP projeckt")

# print()
# print()
# arr = split_sentence("Machine learning is part of data science. A movie theater, cinema, or cinema hall, also known as a movie house, picture house, picture theater or simply theater. Data science is part of the curriculum in many courses. The film is projected with a movie projector onto a large projection screen at the front of the auditorium while the dialogue, sounds and music are played through speakers. Movie theatres stand in a long tradition of theaters that could house all kinds of entertainment. we are doing NLP projeckt")
# print(len(arr))
# labels_arr, prob_arr = clustering(embeddings)

# print(labels_arr)
# print()
# print(prob_arr)