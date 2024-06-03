import copy
import functools
import os
import re
import math
from multiprocessing import Value
from itertools import starmap
import time
import numpy as np
import matplotlib.pyplot as plt

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.nn as nn

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                # device_ids=[dist_util.dev()],
                # output_device=dist_util.dev(),
                device_ids=["cuda:0"],
                output_device="cuda:0",
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        if self.resume_checkpoint != "":
            resume_checkpoint = self.resume_checkpoint
        else:
            resume_checkpoint, self.resume_step = find_resume_checkpoint()

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                trained_model = th.load(resume_checkpoint, map_location=dist_util.dev())
                self.model.load_state_dict(trained_model)

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        if self.resume_checkpoint != "":
            main_checkpoint = self.resume_checkpoint
        else:
            main_checkpoint, self.resume_step = find_resume_checkpoint()
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        if self.resume_checkpoint != "":
            main_checkpoint = self.resume_checkpoint
        else:
            main_checkpoint, self.resume_step = find_resume_checkpoint()
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        print("LR Anneal step: ", self.lr_anneal_steps)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        # print(f"batch allocated memory: {batch.element_size() * batch.nelement()}")
        self.forward_backward(batch, cond)

        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            micro_noise = None
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                noise=micro_noise
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()

def extract_number(f):
    s = re.findall("(\d+).pt",f)
    return (int(s[0]) if s else -1,f)

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    filelistall = os.listdir(get_blob_logdir())
    max_filename = max(filelistall, key=extract_number)
    resume_step = extract_number(max_filename)[0]
    if resume_step == -1 or resume_step == 000000:
        return None, 0
    filename = f"model{(resume_step):06d}.pt"
    path = bf.join(get_blob_logdir(), filename)
    if bf.exists(path):
        return path, resume_step
    return None, 0

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))

        all_x += [th.gather(x, dim=1, index=mask_keep)]
    return th.cat(all_x, dim=0)

def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = th.cat([
        th.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

import time

class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
        visualize=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        self.mask_tmp = th.ones((self.height, self.width), dtype=th.int32)
        self.visualize = visualize

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def convert_to_masked_images(self, images, masks):
        
        def convert_image(image, mask):
            factor = image.shape[-1] / mask.shape[-1]
            upsampler = nn.Upsample(scale_factor=factor, mode='nearest')
            mask = th.ones_like(mask) - mask
            # mask = upsampler(mask[0].float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)).squeeze(0)
            mask = upsampler(mask.float().repeat(3, 1, 1).unsqueeze(0)).squeeze(0)

            # image = image + th.ones_like(image)

            # res = mask * image - th.ones_like(image)
            res = mask * image 

            return res
        # print(f"PrePre masks.shape: {np.shape(masks)}")
        masks = masks[0].unsqueeze(1)
        # print(f"Pre masks.shape: {masks.shape}")
        result = []
        for idx in range(images.size(0)):
            result.append(convert_image(images[idx], masks[idx]))
        # zipped_input = zip(images, masks)
        # processed_res = starmap(convert_image, zipped_input)
        # print(f"list(processed_res): {np.shape(list(processed_res))}")
        # processed_res = th.stack(list(processed_res))
        processed_res = th.stack(result)

        return processed_res

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = th.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
                self.mask_tmp = mask
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = th.randint(0, self.height - h, (1,))
            left = th.randint(0, self.width - w, (1,))
            mask = th.zeros((self.height, self.width), dtype=th.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = th.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        if acceptable_regions is None:
            mask_complement = th.ones((self.height, self.width), dtype=th.int32)
            mask_complement[top:top+h, left:left+w] = 0
        else:
            mask_complement = self.mask_tmp
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        batch_small, batch_big, out_dict = zip(*batch)
        out_dict = out_dict[0]
        # print(out_dict)
        # if type(out_dict) == dict:
        #     print("out_dict is a dict")
        # else:
        #     print("out_dict is a tuple")
        B = len(batch_small)

        collated_batch_small = th.utils.data.default_collate(batch_small)
        collated_batch_big = th.utils.data.default_collate(batch_big)

        seed = self.step()
        g = th.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc, collated_masks_C, collated_masks_eC = [], [], [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)
            collated_masks_C.append(masks_C)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e, masks_C = [], []
            for _ in range(self.nenc):
                mask, mask_C = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                masks_C.append(mask_C)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)
            collated_masks_eC.append(masks_C)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = th.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_C = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_C]
        collated_masks_C = th.utils.data.default_collate(collated_masks_C)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = th.utils.data.default_collate(collated_masks_enc)

        collated_masks_eC = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_eC]
        collated_masks_eC = th.utils.data.default_collate(collated_masks_eC)

        # print(f"Pre collated_batch_small.shape: {collated_batch_small.shape}")
        # print(f"collated_masks_C.shape: {np.shape(collated_masks_C)}")

        collated_batch_small_vis = collated_batch_small
        
        start_time = time.time()
        collated_batch_small = self.convert_to_masked_images(collated_batch_small, collated_masks_C) # small -> diffusion_model
        print("---Image convert time %s seconds ---" % (time.time() - start_time))
        
        # collated_batch_big = self.convert_to_masked_images(collated_batch_big, collated_masks_C)   # big -> ssl_model
        print(f"collated_batch_small.shape: {collated_batch_small.shape}")
        if not self.visualize:
            return collated_batch_small, collated_batch_big, out_dict, collated_masks_enc, collated_masks_pred, collated_masks_C, collated_masks_eC
        else:
            return collated_batch_small, collated_batch_big, out_dict, collated_masks_enc, collated_masks_pred, collated_masks_C, collated_masks_eC, collated_batch_small_vis

def visualize_enc_img(data, mask_eC):
    upsampler = nn.Upsample(scale_factor=16, mode='nearest')
    img = (data[0].permute(1, 2, 0).cpu().numpy() + 1)*127.5
    img = img.astype(np.int64)
    plt.imshow(img)
    
    np_mask_eC = mask_eC.repeat(3, 1, 1).unsqueeze(0).to(th.float64)
    np_mask_eC = upsampler(np_mask_eC).squeeze(0).permute(1, 2, 0)
    print(np_mask_eC.shape)

    np_mask_eC = np_mask_eC.cpu().numpy().astype(np.int64)
    plt.imshow(img*np_mask_eC)
    plt.show()

def visualize_pred_img(data, mask_C):
    upsampler = nn.Upsample(scale_factor=16, mode='nearest')
    img = (data[0].permute(1, 2, 0).cpu().numpy() + 1)*127.5
    img = img.astype(np.int64)
    plt.imshow(img)
    
    mask_C = th.ones_like(mask_C) - mask_C
    np_mask_C = mask_C.repeat(3, 1, 1).unsqueeze(0).to(th.float64)
    np_mask_C = upsampler(np_mask_C).squeeze(0).permute(1, 2, 0)
    print(np_mask_C.shape)

    np_mask_C = np_mask_C.cpu().numpy().astype(np.int64)
    plt.imshow(img*np_mask_C)
    plt.show()
