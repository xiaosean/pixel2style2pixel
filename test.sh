python scripts/inference.py \
--exp_dir=result \
--checkpoint_path=./experiments/20210602_celeb_ir_baseline_from_scratch_v1/checkpoints/best_model.pt \
--data_path=./datasets/simslab_ir2rgb/ir \
--test_batch_size=4 \
--test_workers=4 \
--couple_outputs