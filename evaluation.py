import yaml
import pandas as pd
import matplotlib.pyplot as plt
from InstructorEmbedding import INSTRUCTOR
import umap.umap_ as umap

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def visualize_embeddings(config):
    # 1. Load Data
    df = pd.read_csv(config['data']['path'])
    sentences = df['sentence'].tolist()
    intents = df['intent'].tolist()

    # 2. Compute Embeddings
    model = INSTRUCTOR(config['model']['checkpoint'])
    instruction = config['embedding']['instruction']
    sentences_instructions = [[instruction, sent] for sent in sentences]
    embeddings = model.encode(sentences_instructions)

    # 3. Reduce dimensionality using UMAP
    reducer = umap.UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # 4. Plot the 2D embeddings using UMAP
    plt.figure(figsize=(10, 8))
    for intent in set(intents):
        idx = [i for i, x in enumerate(intents) if x == intent]
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=intent)
    
    plt.legend()
    plt.title("2D UMAP visualization of sentence embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

if __name__ == "__main__":
    config = load_config("config.yml")
    visualize_embeddings(config)
