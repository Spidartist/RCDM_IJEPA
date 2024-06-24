export MODEL_FLAGS_64="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 4"

python run_with_submitit.py --nodes 1 --ngpus 1 --use_volta32 $MODEL_FLAGS_64 $TRAIN_FLAGS --feat_cond --data_dir "/mnt/tuyenld/data/endoscopy/" --type_model ijepa --out_dir checkpoint6 --lr_anneal_steps 350000 --json_dir "/mnt/quanhd/endoscopy/pretrain.json"