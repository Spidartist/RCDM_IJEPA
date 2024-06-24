"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""

import argparse
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from guided_diffusion_rcdm.image_datasets_rc import load_data
from guided_diffusion_rcdm import dist_util, logger
from guided_diffusion_rcdm.get_ssl_models import get_model
from guided_diffusion_rcdm.get_rcdm_models import get_dict_rcdm_model
from guided_diffusion_rcdm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def exclude_bias_and_norm(p):
    return p.ndim == 1

def convert_to_masked_encode_images(images, masks):
    def convert_image(image, mask):
        factor = image.shape[-1] / mask.shape[-1]
        upsampler = nn.Upsample(scale_factor=factor, mode='nearest')
        print("a", mask.shape)
        mask = upsampler(mask.float().repeat(3, 1, 1).unsqueeze(0)).squeeze(0)
        print("b", mask.shape)
        # mask = mask.int()
        print(f"Pre Min: {image.min()}, Max: {image.max()}")
        # image = image + th.ones_like(image)
        print(f"Min: {image.min()}, Max: {image.max()}")
        # res = mask * image - th.ones_like(image)
        res = mask * image 
        print(f"Res Min: {res.min()}, Max: {res.max()}")
        return res
    masks = masks[0].unsqueeze(1)
    result = []
    for idx in range(images.size(0)):
        result.append(convert_image(images[idx], masks[idx]))
    processed_res = th.stack(result)

    return processed_res

def main(args):


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = th.device('cuda:0')
    th.cuda.set_device(device)
    args.gpu = 0
    logger.configure(dir=args.out_dir)

    # tr_normalize = transforms.Normalize(
    #         mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]
    #     )

    # # Crop small
    # transform_zoom = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    #     tr_normalize,
    # ])
    
    # val_dataset_small = datasets.ImageFolder(args.data_dir, transform=transform_zoom)
    # data = DataLoader(
    #     val_dataset_small, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    # )

    logger.log("Load data...")
    data = load_data(
        data_dir=args.data_dir,
        json_dir=args.json_dir,
        batch_size=1,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        random_flip=False,
        visualize=True
    )

    print(f"image_size: {args.image_size}")

    # Use features conditioning
    ssl_model = get_model(args.type_model, args.use_head, reconstruct=True).to(device, non_blocking=True).eval()
    for p in ssl_model.parameters():
        ssl_model.requires_grad = False
    # ssl_dim = ssl_model(th.zeros(1,3,256,256).to(device, non_blocking=True)).size(1)
    # ssl_dim = 1
    ssl_dim = 768
    print(f"SSL dim: {ssl_dim}")

    # ============ preparing data ... ============
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()), G_shared=args.no_shared, feat_cond=True, ssl_dim=ssl_dim
    )

    # Load model
    if args.model_path == "":
        trained_model = get_dict_rcdm_model(args.type_model, args.use_head, reconstruct=True)
    else:
        trained_model = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(trained_model, strict=True)
    # print(model.state_dict)
    model.to(dist_util.dev())
    model.eval()

    # Choose first image
    logger.log("sampling...")
    all_images = []

    sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
    num_current_samples = 0

    while num_current_samples < args.num_images:
        batch_small, batch, cond, masks_enc, masks_pred, masks_C, masks_eC, batch_small_vis = next(data)
        # print(f"masks_eC shape: {np.shape(masks_eC)}")
        # print(f"Batch small shape: {batch_small.shape}")
        # print(f"Batch shape: {batch.shape}")
        batch = batch[0:1].repeat(args.batch_size, 1, 1, 1).to(device, non_blocking=True)
        masks_enc = [u.to(device) for u in masks_enc]
        masks_pred = [u.to(device) for u in masks_pred]
        model_kwargs = {}

        with th.no_grad():
            feat = ssl_model(
                        batch,
                        masks_enc,
                        masks_pred,
                    ).detach()
            model_kwargs["feat"] = feat
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        print(f"Image size: {args.image_size}")
        print(f"Sample shape: {sample.shape}")

        masked_batch_small_vis = convert_to_masked_encode_images(batch_small_vis, masks_eC)

        # Ảnh full
        batch_small_vis = ((batch_small_vis[0:1] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch_small_vis = batch_small_vis.permute(0, 2, 3, 1)
        batch_small_vis = batch_small_vis.contiguous()
        all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in batch_small_vis])

        # Ảnh masked encode
        masked_batch_small_vis = ((masked_batch_small_vis[0:1] + 1) * 127.5).to(th.uint8)
        masked_batch_small_vis = masked_batch_small_vis.permute(0, 2, 3, 1)
        masked_batch_small_vis = masked_batch_small_vis.contiguous()
        all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in masked_batch_small_vis])
        tmp_masked_batch_small_vis = masked_batch_small_vis[0].unsqueeze(0).cpu().numpy()

        # Ảnh masked pred
        batch = ((batch_small[0:1] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch = batch.permute(0, 2, 3, 1)
        batch = batch.contiguous()
        all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in batch])

        # Ảnh gen
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        samples = sample.contiguous()
        all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in samples])
        # all_images.extend([sample.unsqueeze(0).cpu().numpy() + tmp_masked_batch_small_vis for sample in samples])


        logger.log(f"created {len(all_images) * args.batch_size} samples")
        num_current_samples += 1

    arr = np.concatenate(all_images, axis=0)    
    save_image(th.FloatTensor(arr).permute(0,3,1,2), args.out_dir+'/'+args.name+'.jpeg', normalize=True, scale_each=True, nrow=args.batch_size+3, pad_value=255.0)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        json_dir="",
        clip_denoised=True,
        num_images=1,
        batch_size=16,
        use_ddim=False,
        model_path="",
        submitit=False,
        local_rank=0,
        dist_url="env://",
        G_shared=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="samples", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--out_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--no_shared', action='store_false', default=True,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--use_head', action='store_true', default=False,
                        help='Use the projector/head to compute the SSL representation instead of the backbone.')
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--type_model', type=str, default="dino",
                    help='Select the type of model to use.')
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
