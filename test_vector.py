import torch
import pickle


def load_embedding_map(file_path):
    if file_path is None:
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: The file {file_path} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")


if __name__ == "__main__":
    embedding = load_embedding_map(
        "encoder_materials/mscoco_train_qwen2vl_embeddings.pickle")
    
    for x,y in embedding.items():
        print(x, y.shape)
        break
