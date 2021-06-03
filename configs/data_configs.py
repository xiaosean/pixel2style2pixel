from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'ffhq_frontalize': {
		'transforms': transforms_config.FrontalizationTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	# celebs_sketch_to_face': {
	# 	'transforms': transforms_config.SketchToImageTransforms,
	# 	'train_source_root': dataset_paths['celeba_train_sketch'],
	# 	'train_target_root': dataset_paths['celeba_train'],
	# 	'test_source_root': dataset_paths['celeba_test_sketch'],
	# 	'test_target_root': dataset_paths['celeba_test'],
	# },
	'celebs_sketch_to_face': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['fake_face_edge_path'],
		'train_target_root': dataset_paths['fake_face_rgb_path'],
		'test_source_root': dataset_paths['fake_face_edge_path'],
		'test_target_root': dataset_paths['fake_face_rgb_path'],
	},
	'celebs_ir_to_face': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['celebA_ir_path'],
		'train_target_root': dataset_paths['celebA_align_path'],
		'test_source_root': dataset_paths['ir_test'],
		'test_target_root': dataset_paths['ir_2_rgb_test'],
	},
	'celebs_ir_to_face_subset': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['celebA_subset_ir_path'],
		'train_target_root': dataset_paths['celebA_subset_align_path'],
		'test_source_root': dataset_paths['ir_test'],
		'test_target_root': dataset_paths['ir_2_rgb_test'],
	},
	'celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_segmentation'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test_segmentation'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_super_resolution': {
		'transforms': transforms_config.SuperResTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
}
