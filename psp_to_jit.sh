python ./prepare_mobile_model.py \
--exp_dir=result/result_edge_0609_170000 \
--checkpoint_path=./experiments/20210609_celebs_sketch_to_face_from_best/checkpoints/iteration_170000.pt \
--data_path=./datasets/Edge2anime_small_dataset \
--test_batch_size=1 \
--test_workers=4 \
--couple_outputs