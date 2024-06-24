# export CUDA_VISIBLE_DEVICES=2
# export MODEL_FLAGS_256="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# export TRAIN_FLAGS="--lr 1e-4 --batch_size 8"
# export MODEL_FLAGS_128="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# python run_with_submitit_rc.py --nodes 1 --ngpus 1 --use_volta32 $MODEL_FLAGS_128 $TRAIN_FLAGS --feat_cond --data_dir "/home/s/endoscopy/" --type_model ijepa --out_dir checkpointr --lr_anneal_steps 450000 --json_dir "/mnt/quanhd/endoscopy/pretrain.json" --resume_checkpoint "/mnt/quanhd/RCDM_IJEPA/checkpointr/model350000.pt"

export MODEL_FLAGS_64="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 8"

python run_with_submitit_rc.py --nodes 1 --ngpus 1 --use_volta32 $MODEL_FLAGS_64 $TRAIN_FLAGS --feat_cond --data_dir "/mnt/tuyenld/data/endoscopy/" --type_model ijepa --out_dir checkpointr5 --lr_anneal_steps 350000 --json_dir "/mnt/quanhd/endoscopy/pretrain.json"