# 128px 450000 iter van chua cho ra ket qua tot
#export MODEL_FLAGS_128="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"

export MODEL_FLAGS_64="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# python scripts/image_reconstruct.py $MODEL_FLAGS_64 --batch_size 4 --num_images 4 --timestep_respacing 100 --data_dir "/home/s/DATA" --type_model ijepa --out_dir "reconstruct/" --json_dir "/mnt/quanhd/endoscopy/pretrain.json" 
python scripts/image_reconstruct.py $MODEL_FLAGS_64 --batch_size 4 --num_images 4 --timestep_respacing 100 --data_dir "/mnt/tuyenld/data/endoscopy/" --type_model ijepa --out_dir "reconstruct/" --json_dir "/mnt/quanhd/RCDM/test_r.json" 
