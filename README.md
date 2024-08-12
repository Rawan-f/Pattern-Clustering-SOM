# Self-Organizing Maps (SOM) with GloVe Word Embeddings

## Overview

This project implements a Self-Organizing Map (SOM) using the MiniSom library in Python. The SOM is trained on GloVe word embeddings, and the code includes functionality to vary the number of neurons (clusters) and evaluate the clustering performance using precision, recall, and F1 score. 

## Requirements

- Python 3.x
- NumPy
- MiniSom
- scikit-learn
- matplotlib

## Setup

1. Install the required dependencies:

   ```bash
   pip install numpy minisom scikit-learn matplotlib
   ```

2. Download the GloVe word embeddings file (`glove.6B.50d.txt`) and place it in the project directory.

3. Prepare the dataset files (`animals.txt`, `countries.txt`, `fruits.txt`, `veggies.txt`) and place them in the project directory.

## Usage

Run the main script to train and evaluate the SOM for varying numbers of neurons:

```bash
python task4.1.py
```
```bash
python task4.2.py
```
## Results

The script generates a plot showing the performance metrics (precision, recall, and F1 score) for different numbers of neurons.
