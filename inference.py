import os
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from InstructorEmbedding import INSTRUCTOR

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def infer_intent_and_answer(user_input, config):
    model_save_dir = config['model']['save_path']
    embeddings = np.load(os.path.join(model_save_dir, 'embeddings.npy'))
    intent_answer_df = pd.read_csv(os.path.join(model_save_dir, 'intent_answer_mapping.csv'))

    model = INSTRUCTOR(config['model']['checkpoint'])
    instruction = config['embedding']['instruction']
    user_embedding = model.encode([[instruction, user_input]])

    similarities = cosine_similarity(user_embedding, embeddings)
    closest_index = np.argmax(similarities)

    intent = intent_answer_df.iloc[closest_index]['intent']
    answer = intent_answer_df.iloc[closest_index]['answer']
    
    return intent, answer

if __name__ == "__main__":
    config = load_config("config.yml")

    print("Chatbot is ready! Type 'exit' or 'bye' to end the conversation.")
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'bye']:
            print("Chatbot: Goodbye!")
            break

        intent, answer = infer_intent_and_answer(user_input, config)
        print(f"Chatbot: {answer}")
