python scripts/train.py \
--dataset_type=celebs_sketch_to_face \
--exp_dir=./experiments/20210620_amp_celebs_sketch_from_sketch \
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
--input_nc=1
# --checkpoint_path=./experiments/20210609_celebs_sketch_to_face_from_best/checkpoints/iteration_170000.pt
# --checkpoint_path=./pretrained_models/psp_celebs_sketch_to_face.pt \
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


# self.opts = Namespace(batch_size=8, board_interval=50, checkpoint_path=None, 
# dataset_type='celebs_sketch_to_face', device='cuda:0', 
# encoder_type='GradualStyleEncoder', 
# exp_dir='./experiments/20210620_amp_celebs_sketch_from_sketch', 
# id_lambda=0.0, image_interval=100, input_nc=1, l2_lambda=1.0, 
# l2_lambda_crop=0, label_nc=1, learn_in_w=False, learning_rate=0.0001, 
# lpips_lambda=0.8, lpips_lambda_crop=0, max_steps=500000, moco_lambda=0, 
# n_styles=18, optim_name='ranger', output_size=1024, resize_factors=None, 
# save_interval=5000, start_from_latent_avg=True, 
# stylegan_weights='pretrained_models/stylegan2-ffhq-config-f.pt', test_batch_size=8, 
# test_workers=8, train_decoder=False, val_interval=2500, w_norm_lambda=0.005, workers=20)
