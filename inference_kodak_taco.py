import os
import sys
import json
import math
import torch
import shutil
import pandas as pd
import torchvision
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from transformers import CLIPTextModel, AutoTokenizer
from pytorch_msssim import ms_ssim as ms_ssim_func
import lpips

from models import TACO
from config.config import model_config
from utils.utils import *


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

loss_fn_alex = lpips.LPIPS(net='alex').to(device)
loss_fn_alex.requires_grad_(False)


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def parse_args_for_inference(argv):
    parser = argparse.ArgumentParser(description="Kodak validation script.")
    parser.add_argument("--image_folder_root", type=str,
                        default='./kodak', help="Path to Kodak images")
    parser.add_argument("--checkpoint", type=str,
                        default='./checkpoint/lambda_0.015.pth.tar', help="Checkpoint path")
    parser.add_argument("--save_path", type=str,
                        default='./compression_kodak_results', help="Path to save results")
    parser.add_argument("--caption_json", type=str,
                        default='./materials/kodak_ofa.json', help="JSON with captions")
    parser.add_argument("--clip_model_name", type=str,
                        default="./clip/clip-vit-base-patch32", help="CLIP path")
    return parser.parse_args(argv)


def setup_model_and_tokenizer(clip_model_name):
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    clip_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    clip_model.requires_grad_(False)

    config = model_config()
    model = TACO(config, text_embedding_dim=clip_model.config.hidden_size).to(device)
    return model, clip_model, tokenizer


def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    try:
        model.load_state_dict(state_dict)
    except:
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(cleaned)
        except:
            model.module.load_state_dict(cleaned)
    model.requires_grad_(False)
    model.update()


def prepare_save_folders(save_path, checkpoint_path):
    tag = os.path.basename(checkpoint_path)[:-8]
    folder = os.path.join(save_path, tag)
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(os.path.join(folder, 'figures'))
    os.makedirs(os.path.join(folder, 'temp'))
    return folder


def run_single_image_compression(model, clip_model, tokenizer, img_tensor, caption):
    x = img_tensor.unsqueeze(0).to(device)
    _, _, H, W = x.shape

    pad_h = (64 - H % 64) if H % 64 != 0 else 0
    pad_w = (64 - W % 64) if W % 64 != 0 else 0
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

    clip_token = tokenizer([caption], padding="max_length", max_length=38,
                           truncation=True, return_tensors="pt").to(device)
    text_embeddings = clip_model(**clip_token).last_hidden_state

    out_enc = model.compress(x_padded, text_embeddings)
    shape = out_enc["shape"]

    return x, out_enc, shape, text_embeddings, (H, W)


def run_inference_on_dataset(args, model, clip_model, tokenizer):
    with open(args.caption_json, 'r') as f:
        image_caption_pairs = json.load(f)

    save_folder = prepare_save_folders(args.save_path, args.checkpoint)
    stat_csv = {'image_name': [], 'bpp': [],
                'psnr': [], 'ms_ssim': [], 'lpips': []}
    mean_stats = {'bpp': 0.0, 'psnr': 0.0, 'ms_ssim': 0.0, 'lpips': 0.0}

    for img_name, caption in tqdm(image_caption_pairs.items(), desc="compress:"):
        img_path = os.path.join(args.image_folder_root, img_name)
        img = torchvision.transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        x, out_enc, shape, text_embeddings, (H, W) = run_single_image_compression(
            model, clip_model, tokenizer, img, caption)

        bin_path = os.path.join(save_folder, "temp", img_name)
        with Path(bin_path).open("wb") as f:
            write_uints(f, (H, W))
            write_body(f, shape, out_enc["strings"])
        bpp = filesize(bin_path) * 8 / (H * W)

        with Path(bin_path).open("rb") as f:
            original_size = read_uints(f, 2)
            strings, shape = read_body(f)

        out = model.decompress(strings, shape, text_embeddings)
        x_hat = out["x_hat"][:, :, :original_size[0], :original_size[1]]

        psnr = compute_psnr(x, x_hat)
        try:
            ms_ssim = ms_ssim_func(x, x_hat, data_range=1.).item()
        except:
            ms_ssim = ms_ssim_func(torchvision.transforms.Resize(256)(
                x), torchvision.transforms.Resize(256)(x_hat), data_range=1.).item()
        lpips_score = loss_fn_alex(x, x_hat).item()

        torchvision.utils.save_image(
            x_hat, os.path.join(save_folder, 'figures', img_name))

        stat_csv['image_name'].append(img_name)
        stat_csv['bpp'].append(bpp)
        stat_csv['psnr'].append(psnr)
        stat_csv['ms_ssim'].append(ms_ssim)
        stat_csv['lpips'].append(lpips_score)

        mean_stats['bpp'] += bpp
        mean_stats['psnr'] += psnr
        mean_stats['ms_ssim'] += ms_ssim
        mean_stats['lpips'] += lpips_score

    shutil.rmtree(os.path.join(save_folder, "temp"))
    return stat_csv, mean_stats, save_folder


def save_and_log_results(stat_csv, mean_stats, save_folder):
    df = pd.DataFrame(stat_csv)
    df.to_csv(os.path.join(save_folder, "stat_per_image.csv"), index=False)

    n = len(stat_csv['image_name'])
    for key in mean_stats:
        mean_stats[key] /= n

    with open(os.path.join(save_folder, "mean_stat.json"), "w") as f:
        json.dump(mean_stats, f, indent=4)

    print(
        f"\nBPP: {mean_stats['bpp']:.4f}, PSNR: {mean_stats['psnr']:.4f}, MS-SSIM: {mean_stats['ms_ssim']:.4f}, LPIPS: {mean_stats['lpips']:.4f}")


def main(argv):
    args = parse_args_for_inference(argv)
    model, clip_model, tokenizer = setup_model_and_tokenizer(
        args.clip_model_name)
    load_checkpoint(model, args.checkpoint)
    stat_csv, mean_stats, save_folder = run_inference_on_dataset(
        args, model, clip_model, tokenizer)
    save_and_log_results(stat_csv, mean_stats, save_folder)


if __name__ == "__main__":
    main(sys.argv[1:])
