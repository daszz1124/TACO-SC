import torch
from PIL import Image
from src.arguments import ModelArguments
from mmeb_model import loading_model, loading_processor, process_embedding

if __name__ == "__main__":

    image_path = '/home/iisc/zsd/project/VG2SC/VLM2Vec/figures/example.jpg'
    query_text = "Represent the given image with the following question: What is in the image"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = ModelArguments(
        model_name="/home/iisc/zsd/project/VG2SC/MMEB-Models/Qwen/Qwen2-VL-2B-Instruct",
        checkpoint_path="/home/iisc/zsd/project/VG2SC/MMEB-Models/VLM2Vec/VLM2Vec-Qwen2VL-2B",
        pooling="last",
        normalize=True,
        model_backbone="qwen2_vl",
        lora=True
    )

    qry_embeddings = process_embedding(
        model_args=model_args,
        image_path=image_path,
        query_text=query_text,
        device=device,
    )

    print("Query Embedding Shape:", qry_embeddings.shape)