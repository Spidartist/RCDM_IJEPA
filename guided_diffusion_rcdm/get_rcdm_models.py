# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import torch
import torch.nn as nn
from torchvision import models as torchvision_models

def get_dict_rcdm_model(model="dino", use_head=False, reconstruct=False):
    '''
    Download checkpoints of RCDM.
    '''

    if model == "supervised":
        trained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_supervised.pt", map_location="cpu")
        return trained_model

    elif model == "simclr":
        if use_head:
            trained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_simclr_head.pt", map_location="cpu")
        else:
            trained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_simclr_trunk.pt", map_location="cpu")
        return trained_model

    elif model == "barlow":
        if use_head:
            trained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_barlow_head.pt", map_location="cpu")
        else:
            trained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_barlow_trunk.pt", map_location="cpu")
        return trained_model

    elif model == "vicreg":
        if use_head:
            trained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_vicreg_head.pt", map_location="cpu")
        else:
            trained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_vicreg_trunk.pt", map_location="cpu")
        return trained_model

    elif model == "dino":
        if use_head:
            trained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_dino_head.pt", map_location="cpu")
        else:
            trained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_dino_trunk.pt", map_location="cpu")
        return trained_model
    elif model == "ijepa":
        if not reconstruct:
            trained_model = torch.load("/mnt/quanhd/RCDM/checkpoint4/model130000.pt")
        else:
            trained_model = torch.load("/mnt/quanhd/RCDM/checkpointr5/ema_0.9999_100000.pt")
        return trained_model

    else:
        print("No model found")
        exit(1)
