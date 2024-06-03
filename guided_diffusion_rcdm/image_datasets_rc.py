import math
import random
import json
from PIL import Image
import os
import blobfile as bf
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets.folder import find_classes
from .dist_util import get_rank, get_world_size
from torchvision import datasets, transforms
from .train_util import MaskCollator as MBMaskCollator

def load_data(
    *,
    data_dir,
    json_dir,
    batch_size,
    image_size,
    rank=0,
    world_size=1,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    visualize=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.
    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    all_files = _get_img_paths_from_json(json_path=json_dir, data_dir=data_dir)
    print(f"Read data from json file {json_dir} completed!")

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        #class_names = [bf.basename(path).split("_")[0] for path in all_files]
        class_names = [bf.dirname(path) for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=rank,
        num_shards=world_size,
        random_crop=random_crop,
        random_flip=random_flip,
    )

    mask_collator = MBMaskCollator(
        input_size=(256, 256),
        patch_size=16,
        pred_mask_scale=(0.15, 0.2),
        enc_mask_scale=(0.85, 1),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        allow_overlap=False,
        min_keep=10,
        visualize=visualize
    )

    #return dataset
    if deterministic:
        loader = DataLoader(
            dataset, collate_fn=mask_collator, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, collate_fn=mask_collator, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    # return loader
    while True:
        yield from loader

def process_path(old_path, data_dir):
    old_ls = old_path.split("/")
    new_ls = []
    for idx, elem in enumerate(old_ls):
        if idx < len(old_ls) - 1:
            new_elem = elem.replace(" ", "_")
        else:
            new_elem = elem
        new_ls.append(new_elem)
    new_path = "/".join(new_ls)
    new_path = os.path.join(data_dir, new_path)
    return new_path

def _get_img_paths_from_json(json_path, data_dir):
    f = open(json_path)
    data = json.load(f)
    img_paths = [os.path.join(data_dir, elem) for elem in data["train"]]
    return img_paths


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def load_single_image(path):
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    # arr = np.array(pil_image)
    arr = center_crop_arr(pil_image, 256)
    # arr = arr.astype(np.float32) / 127.5 - 1
    arr = arr.astype(np.float32) / 255
    arr = np.transpose(arr, [2, 0, 1])
    return arr
    
class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        # print(path)
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
            arr2 = random_crop_arr(pil_image, 256)
        else:
            arr = center_crop_arr(pil_image, self.resolution)
            arr2 = center_crop_arr(pil_image, 256)
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            arr2 = arr2[:, ::-1]
        # arr = arr.astype(np.float32) / 127.5 - 1
        # arr2 = arr2.astype(np.float32) / 127.5 - 1
        arr = arr.astype(np.float32) / 255
        arr2 = arr2.astype(np.float32) / 255

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        # We return two images, one of size 224x224 for the SSL model and one for 
        # the generative model with a specific size
        # batch_small, batch, cond
        return np.transpose(arr, [2, 0, 1]), np.transpose(arr2, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

if __name__ == "__main__":
    print(_list_image_files_recursively("/mnt/ducntm/endoscopy/DATA/"))
    