python emotion_generator_pipeline.py \
--exp_dir=result_0616_simslab \
--checkpoint_path=./experiments/20210603_celeb_ir_baseline_from_0602_60k/checkpoints/best_model.pt \
--data_path=./datasets/Edge2anime_small_dataset \
--test_batch_size=8 \
--test_workers=4 \
--couple_outputs
# --mix_alpha=0.5
# --latent_mask=8,9,10,11,12,13,14,15,16,17 \
# --mix_alpha=0.5 \