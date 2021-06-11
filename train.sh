python scripts/train.py \
--dataset_type=celebs_sketch_to_face \
--exp_dir=./experiments/20210609_celebs_sketch_to_face_from_best \
--workers=20 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0 \
--w_norm_lambda=0.005 \
--label_nc=1 \
--input_nc=1 \
--checkpoint_path=./pretrained_models/psp_celebs_sketch_to_face.pt \
# --checkpoint_path=./experiments/20210603_celeb_ir_baseline_from_0602_60k/checkpoints/iteration_120000.pt
# --exp_dir=./experiments/20210605_celeb_sketch_baseline_from_0603best \
# --checkpoint_path=./pretrained_models/psp_celebs_sketch_to_face.pt
# --checkpoint_path=./experiments/20210602_celeb_ir_baseline_from_scratch_v1/checkpoints/iteration_60000.pt

# --dataset_type=celebs_ir_to_face \
# --checkpoint_path=./pretrained_models/psp_celebs_sketch_to_face.pt

# TODO:Zoom data aug

# CUDA_LAUNCH_BLOCKING=1 python scripts/train.py \
# --dataset_type=celebs_ir_to_face \
# --exp_dir=./experiments/20210526_celeb_ir_baseline \
# --workers=20 \
# --batch_size=8 \
# --test_batch_size=8 \
# --test_workers=8 \
# --val_interval=2500 \
# --save_interval=5000 \
# --encoder_type=GradualStyleEncoder \
# --start_from_latent_avg \
# --lpips_lambda=0.8 \
# --l2_lambda=1 \
# --id_lambda=0 \
# --w_norm_lambda=0.005 \
# --label_nc=1 \
# --input_nc=3

