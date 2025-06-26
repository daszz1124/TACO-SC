from mmeb_model import loading_processor, model_args
from src.model_utils import get_backbone_name, load_processor, QWEN2_VL, vlm_image_tokens
from src.model_utils import PHI3V, vlm_image_tokens
from torchvision import transforms
import PIL.Image as Image
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from torch.utils.data import DataLoader

import torch
import os

from mmeb_model import loading_processor, model_args, loading_model
import pickle

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch


def Qwen2_VL_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text=[
                               text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(images=[image], text=[
                               text], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad(
        {'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'texts': texts,
        'images': images,
    }
    if image_exists:
        pixel_value_shape_for_padding = list(
            v.shape for v in pixel_values if v is not None)[0]
        pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(
            pixel_value_shape_for_padding) for v in pixel_values]
        pixel_values = torch.stack(pixel_values, dim=0)
        inputs['pixel_values'] = pixel_values
        inputs['image_grid_thw'] = image_grid_thw
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_grid_thw'] = [None] * input_ids.shape[0]

    return inputs


def collate_fn_qwen2vl(batch, processor, max_length=None):
    """
    batch: list of (img_path:str, caption:str)
    """
    img, captions, img_path, text = zip(*batch)

    model_inputs = {
        "text": list(captions),
        "image": img,
    }

    return img_path, text, Qwen2_VL_process_fn(model_inputs, processor, max_length=max_length)


class MSCOCO2MMEBDataset(Dataset):
    def __init__(self, dataset_folder, image_size=(256, 256)):
        self.dataset_folder = dataset_folder
        image_list_path = './materials/mscoco_train_name_list_larger_than_256.json'
        caption_list_path = './materials/mscoco_train_caption_list_larger_than_256.json'

        with open(image_list_path, 'r') as f:
            self.image_name_list = json.load(f)
        with open(caption_list_path, 'r') as f:
            self.caption_list = json.load(f)

        if len(self.image_name_list) != len(self.caption_list):
            raise ValueError("Mismatch between number of images and captions")

        self.paired_data = [
            {"text": caption, "img_path": image_path}
            for caption, image_path in zip(self.caption_list, self.image_name_list)
        ]

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        item = self.paired_data[idx]
        img_path = os.path.join(self.dataset_folder,
                                "train2014", item["img_path"])

        return self._get_image(img_path), f'{vlm_image_tokens[QWEN2_VL]} {item["text"]}', img_path, item["text"]

    def _process_image(self, image, resolution="mid"):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        image = Image.open(img_path)
        return self._process_image(image)

    @property
    def get_paired_data(self):
        return self.paired_data


class KodakDataset(Dataset):
    def __init__(self, dataset_folder='/home/minkyu4506/Dataset/Kodak', image_size=(256, 256)):
        self.dataset_folder = dataset_folder
        self.image_size = image_size
        
        self.caption_file = os.path.join("/home/iisc/zsd/project/VG2SC/TACO/materials", "kodak_ofa.json")
        self._load_data()
        assert len(self.image_name_list) == len(
            self.caption_list), "图像和caption长度不一致"

        # 创建配对数据
        self.paired_data = [{"text": t, "img_path": p}
                           for t, p in zip(self.caption_list, self.image_name_list)]

    def _load_data(self):
        """加载Kodak数据集的图像路径和描述"""
        try:
            with open(self.caption_file, 'r') as f:
                captions_dict = json.load(f)
            self.image_name_list = list(captions_dict.keys())
            self.caption_list = list(captions_dict.values())
            
        except Exception as e:
            print(f"加载数据集时出错: {e}")
            # 设置空列表作为回退
            self.image_name_list = []
            self.caption_list = []

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        item = self.paired_data[idx]
        img_path = os.path.join(self.dataset_folder, item["img_path"])

        return self._get_image(img_path), f'{vlm_image_tokens[QWEN2_VL]} {item["text"]}', img_path, item["text"]

    def _process_image(self, image, resolution="mid"):
        """处理图像，调整大小"""
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        """获取并处理图像"""
        if img_path == "":
            return None
        try:
            image = Image.open(img_path).convert('RGB')
            return self._process_image(image)
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            return None

    @property
    def get_paired_data(self):
        return self.paired_data    

def build_qwen2vl_dataloader(processor, datasets, batch_size=64, num_workers=4, max_length=4096):

    loader = DataLoader(
        datasets,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn_qwen2vl(
            batch, processor, max_length=max_length),
        pin_memory=True
    )

    return loader


if __name__ == "__main__":
    processor = loading_processor(model_args)

    image_size = (256, 256)

    # eval_qry_dataset = MSCOCO2MMEBDataset(
    #     dataset_folder='/home/iisc/zsd/project/VG2SC/datasets/MSCOCO',
    #     image_size=image_size,
    # )
    eval_qry_dataset = KodakDataset(
        dataset_folder="/home/iisc/zsd/project/VG2SC/TACO/kodak",
        image_size=image_size
    )

    eval_loader = build_qwen2vl_dataloader(
        processor, eval_qry_dataset, batch_size=128, num_workers=4, max_length=4096)

    # enccode_qry_path = './encoder_materials/mscoco_train_qwen2vl_embeddings.pickle'
    encode_qry_path = './encoder_materials/kodak_val_qwen2vl_embeddings.pickle'

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    model = loading_model(model_args, device=device)

    embedding_map = {}
    with torch.no_grad():
        for image_paths, query_texts, batch in tqdm(eval_loader, desc="Encode query"):
            batch = batch_to_device(batch, device)
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                output = model(qry=batch)
            encoded_tensor = output["qry_reps"].cpu().detach().float().numpy()

            for img_path, caption, embedding in zip(image_paths, query_texts, encoded_tensor):
                key = (caption, img_path)
                embedding_map[key] = embedding

    with open(encode_qry_path, 'wb') as f:
        pickle.dump(embedding_map, f)
