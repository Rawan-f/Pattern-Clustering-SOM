import numpy as np
from minisom import MiniSom

# Load GloVe word embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip().split('\n')

glove_embeddings = load_glove_embeddings('glove.6B.50d.txt')
animal_data = load_dataset('animals.txt')
country_data = load_dataset('countries.txt')
fruit_data = load_dataset('fruits.txt')
veggie_data = load_dataset('veggies.txt')

all_data = animal_data + country_data + fruit_data + veggie_data

word_embeddings = {word: glove_embeddings[word] for word in all_data if word in glove_embeddings}

# Implement SOM algorithm with MiniSom
def train_som(word_embeddings, k, epochs=500):
    som_model = MiniSom(k, 1, input_len=50, sigma=0.5, learning_rate=0.5)
    som_model.random_weights_init(np.array(list(word_embeddings.values())))
    som_model.train_random(np.array(list(word_embeddings.values())), epochs, verbose=False)
    return som_model

# Assign each word to a cluster
def assign_clusters(som_model, word_embeddings):
    predicted_labels = np.argmin(
        np.linalg.norm(som_model._weights - np.array(list(word_embeddings.values())), axis=2),
        axis=0
    )
    return predicted_labels

som_model = train_som(word_embeddings, 2)