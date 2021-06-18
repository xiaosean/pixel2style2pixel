python scripts/inference.py \
--exp_dir=result/result_edge_paper \
--checkpoint_path=./pretrained_models/psp_celebs_sketch_to_face.pt \
--data_path=./datasets/Edge2anime_small_dataset \
--test_batch_size=1 \
--test_workers=4 \
--couple_outputs


# python scripts/inference.py \
# --exp_dir=result_0609_edge \
# --checkpoint_path=./experiments/20210609_celebs_sketch_to_face_from_best/checkpoints/best_model.pt \
# --data_path=./datasets/Edge2anime_small_dataset \
# --test_batch_size=1 \
# --test_workers=4 \
# --couple_outputs

# psp_celebs_sketch_to_face.pt
# --checkpoint_path=./experiments/20210605_celeb_sketch_baseline_from_0603best/checkpoints/best_model.pt \
# --mix_alpha=0.5 \
# --latent_mask=0,1,2,3 \
# --latent_mask=0,1,2,3,4,5,6,7 \
# --latent_mask=8,9,10,11,12,13,14,15,16,17 \

# --latent_mask=8,9,10,11,12,13,14,15,16,17 \
# --mix_alpha=0.75 \

# python scripts/inference.py \
# --exp_dir=result/result_edge_own \
# --checkpoint_path=./experiments/20210609_celebs_sketch_to_face_from_best/checkpoints/best_model.pt \
# --data_path=./datasets/Edge2anime_small_dataset \
# --test_batch_size=1 \
# --test_workers=4 \
# --couple_outputs

