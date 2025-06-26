from mmeb_model import loading_processor, model_args
from src.model_utils import get_backbone_name, load_processor, QWEN2_VL, vlm_image_tokens
from transformers import AutoTokenizer, ViTImageProcessor
from torchvision import transforms
import PIL.Image as Image
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
import torch

import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MSCOCO_train_dataset(Dataset):
    def __init__(self, dataset_folder='/home/minkyu4506/Dataset/MSCOCO', image_size=(256, 256), clip_name="openai/clip-vit-base-patch32", node_rank=0):

        self.dataset_folder = dataset_folder

        self.tokenizer = AutoTokenizer.from_pretrained(clip_name)

        with open('./materials/mscoco_train_name_list_larger_than_256.json', 'r') as f:
            self.image_name_list = json.load(f)
        with open('./materials/mscoco_train_caption_list_larger_than_256.json', 'r') as f:
            self.caption_list = json.load(f)

        self.transform = transforms.Compose(
            [transforms.RandomCrop(image_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.image_name_list)

    def load_image(self, image_path):

        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return self.transform(image)

    def __getitem__(self, idx):

        img_name = self.image_name_list[idx]

        img = self.load_image(f'{self.dataset_folder}/train2014/{img_name}')
        caption = self.caption_list[idx]

        tokenized_output = self.tokenizer(
            caption, padding="max_length", max_length=38, truncation=True, return_tensors="pt")

        token = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']

        if len(token.size()) != 1:
            token = token.squeeze(0)

        if len(attention_mask.size()) != 1:
            attention_mask = attention_mask.squeeze(0)

        return img, token, attention_mask


class MSCOCOMMEB_TrainDataset(Dataset):
    def __init__(
        self,
        dataset_folder: str = None,
        image_path: str = None,
        caption_path: str = None,
        embedding_file_path: str = None,
        image_size=(256, 256),
    ):

        self.dataset_folder = dataset_folder
        self.processor = loading_processor(model_args)

        with open(image_path, 'r') as f:
            self.image_name_list = json.load(f)
        with open(caption_path, 'r') as f:
            self.caption_list = json.load(f)

        self.transform = transforms.Compose(
            [transforms.RandomCrop(image_size), transforms.ToTensor()]
        )

        self.embedding_map = self.load_embedding_map(embedding_file_path)

    def load_embedding_map(self, file_path):
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

    def __len__(self):
        return len(self.image_name_list)

    def load_image(self, image_path):

        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return self.transform(image)

    def __getitem__(self, idx):

        img_name = self.image_name_list[idx]
        img_name = f'{self.dataset_folder}/train2014/{img_name}'
        img = self.load_image(img_name)
        caption = self.caption_list[idx]
        embedding_vector = self.embedding_map[(caption, img_name)]
        embedding_vector = torch.from_numpy(
            embedding_vector).unsqueeze(0).float()

        return img, embedding_vector
