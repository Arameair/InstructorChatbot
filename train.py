import yaml
import pandas as pd
import numpy as np
import os
from InstructorEmbedding import INSTRUCTOR

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_and_save_embeddings(config):
    # Ensure the model_save_dir exists
    model_save_dir = config['model']['save_path']
    os.makedirs(model_save_dir, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(config['data']['path'])
    sentences = df['sentence'].tolist()

    # 2. Compute Embeddings
    model = INSTRUCTOR(config['model']['checkpoint'])
    instruction = config['embedding']['instruction']
    sentences_instructions = [[instruction, sent] for sent in sentences]
    embeddings = model.encode(sentences_instructions)

    # 3. Save Embeddings and Intents
    np.save(os.path.join(model_save_dir, 'embeddings.npy'), embeddings)
    df[['intent', 'answer']].to_csv(os.path.join(model_save_dir, 'intent_answer_mapping.csv'), index=False)

    print(f"Embeddings and intent mappings saved to {model_save_dir}")

if __name__ == "__main__":
    config = load_config("config.yml")
    train_and_save_embeddings(config)
