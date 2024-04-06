import os
os.environ["KERAS_BACKEND"] = "jax"
import torch

from clustering import sent_embedding
from clustering import clustering
from clustering import split_sentence

# import numpy as np

def sorter(labels_arr, text_split):
    sorted_input = []
    for i in range(len(labels_arr)):
        sorted_input.append("")
    for i in range(len(labels_arr)):
        sorted_input[labels_arr[i]] += text_split[i] + " "
    sorted_input = list(filter(None, sorted_input))

    return sorted_input

model = torch.load("./pretrained/model_50_epochs.pth")

text_input = "Machine learning is part of data science. A movie theater, cinema, or cinema hall, also known as a movie house, picture house, picture theater or simply theater. Data science is part of the curriculum in many courses. The film is projected with a movie projector onto a large projection screen at the front of the auditorium while the dialogue, sounds and music are played through speakers. Movie theatres stand in a long tradition of theaters that could house all kinds of entertainment. We are doing NLP project"

text_split = split_sentence(text_input)
embeddings = sent_embedding(text_input)
labels_arr, prob_arr = clustering(embeddings)

sorted_input = sorter(labels_arr, text_split)

generated_list = []
for i in sorted_input:
    generated_list.append(model.predict([i]))

print(generated_list)


