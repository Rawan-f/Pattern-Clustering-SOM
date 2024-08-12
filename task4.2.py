import numpy as np
from minisom import MiniSom
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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

#ground truth labels for the instances
ground_truth_labels = ['animal'] * len(animal_data) + ['country'] * len(country_data) + ['fruit'] * len(fruit_data) + ['vegetarian'] * len(veggie_data)

# Convert ground truth labels to numerical format for evaluation
label_to_number = {'animal': 0, 'country': 1, 'fruit': 2, 'vegetarian': 3}
true_labels = [label_to_number[label] for label in ground_truth_labels]

def train_evaluate_som(word_embeddings, k_range):
    precision_scores, recall_scores, f1_scores = [], [], []

    for k in k_range:
        # Train SOM
        som_model = MiniSom(k, 1, input_len=50, sigma=0.5, learning_rate=0.5, random_seed=42)
        som_model.random_weights_init(np.array(list(word_embeddings.values())))
        som_model.train_random(np.array(list(word_embeddings.values())), 500, verbose=False)
        predicted_labels = np.argmin(
            np.linalg.norm(som_model._weights - np.array(list(word_embeddings.values())), axis=2),
            axis=0
        )

        true_labels_adjusted = true_labels[:len(predicted_labels)]

        dominant_labels = []
        for cluster in range(k):
            cluster_mask = (predicted_labels == cluster)
            cluster_labels = np.array(true_labels_adjusted)[cluster_mask]
            dominant_label = np.argmax(np.bincount(cluster_labels))
            dominant_labels.append(dominant_label)

        mapped_labels = np.array([dominant_labels[label] for label in predicted_labels])

        # Compute precision, recall, and F1 score
        precision = precision_score(true_labels_adjusted, mapped_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels_adjusted, mapped_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels_adjusted, mapped_labels, average='weighted', zero_division=0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return precision_scores, recall_scores, f1_scores

# Define the range of k (neurons)
k_range = range(2, 11)

# Train and evaluate SOM for each value of k
precision_scores, recall_scores, f1_scores = train_evaluate_som(word_embeddings, k_range)


plt.figure(figsize=(10, 6))
plt.plot(k_range, precision_scores, label='Precision', marker='o')
plt.plot(k_range, recall_scores, label='Recall', marker='o')
plt.plot(k_range, f1_scores, label='F1 Score', marker='o')
plt.xlabel('Number of Neurons (k)')
plt.ylabel('Score')
plt.title('Performance Metrics for Varying Neurons in SOM')
plt.legend()
plt.grid(True)
plt.show()