

import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.dataset import EvalDataset
from src.collator import EvalCollator
from src.utils import print_rank
from src.model_utils import get_backbone_name, load_processor, QWEN2_VL, vlm_image_tokens


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch


def loading_processor(model_args):
    return load_processor(model_args)


def loading_model(model_args, device=None):
    """Load and prepare the model."""
    model = MMEBModel.load(model_args)
    model.to(device, dtype=torch.bfloat16)
    model.eval()
    return model


def process_embedding(model_args, image_path, query_text, device):

    model = loading_model(model_args, device)
    processor = loading_processor(model_args)

    query_inputs = processor(
        text=f'{vlm_image_tokens[QWEN2_VL]} {query_text}',
        images=Image.open(image_path),
        return_tensors="pt"
    )
    query_inputs = {key: value.to(device)
                    for key, value in query_inputs.items()}
    query_inputs['pixel_values'] = query_inputs['pixel_values'].unsqueeze(0)
    query_inputs['image_grid_thw'] = query_inputs['image_grid_thw'].unsqueeze(
        0)

    with torch.no_grad():
        query_reps = model(qry=query_inputs)["qry_reps"]
        return query_reps


# def process_batch_embedding_with_model(model, processor, img_names, captions, device):

#     texts = [f"{vlm_image_tokens[QWEN2_VL]} {caption}" for caption in captions]
#     images = [Image.open(img_name) for img_name in img_names]

#     query_inputs = processor(
#         text=texts,
#         images=images,
#         return_tensors="pt",
#         padding=True,
#         truncation=True
#     )
#     query_inputs = batch_to_device(query_inputs, device)
#     query_reps = model(qry=query_inputs)["qry_reps"]
#     return query_reps

def process_batch_embedding_with_model(model, processor, img_names, captions, device):
    querys_list = []
    for img,caption in zip(img_names, captions):
        query_inputs = processor(
            text=f'{vlm_image_tokens[QWEN2_VL]} {caption}',
            images=Image.open(img),
            return_tensors="pt"
        )
        query_inputs = {key: value.to(device)
                        for key, value in query_inputs.items()}
        query_inputs['pixel_values'] = query_inputs['pixel_values'].unsqueeze(0)
        query_inputs['image_grid_thw'] = query_inputs['image_grid_thw'].unsqueeze(
            0)

        with torch.no_grad():
            query_reps = model(qry=query_inputs)["qry_reps"]
            querys_list.append(query_reps)
    return torch.stack(querys_list, dim=0).float()

def process_embedding_with_model(model, processor, img_name, caption, device):
    query_inputs = processor(
        text=f'{vlm_image_tokens[QWEN2_VL]} {caption}',
        images=Image.open(img_name),
        return_tensors="pt"
    )
    query_inputs = {key: value.to(device)
                    for key, value in query_inputs.items()}
    query_inputs['pixel_values'] = query_inputs['pixel_values'].unsqueeze(0)
    query_inputs['image_grid_thw'] = query_inputs['image_grid_thw'].unsqueeze(
        0)

    with torch.no_grad():
        query_reps = model(qry=query_inputs)["qry_reps"]
        return query_reps


model_args = ModelArguments(
    model_name="/home/iisc/zsd/project/VG2SC/MMEB-Models/Qwen/Qwen2-VL-2B-Instruct",
    checkpoint_path="/home/iisc/zsd/project/VG2SC/MMEB-Models/VLM2Vec/VLM2Vec-Qwen2VL-2B",
    pooling="last",
    normalize=True,
    model_backbone="qwen2_vl",
    lora=True
)
