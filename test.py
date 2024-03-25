from guided_diffusion_rcdm.image_datasets import _list_image_files_recursively, _get_img_paths_from_json
from pprint import pprint

res = _list_image_files_recursively("/mnt/ducntm/endoscopy/DATA/")

all_files = _get_img_paths_from_json(json_path="/mnt/tuyenld/mae/data_annotation/pretrain.json", data_dir="/mnt/ducntm/endoscopy/DATA/")

print(all_files[0])
print(len(all_files))