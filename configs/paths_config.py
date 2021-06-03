dataset_paths = {
	'celeba_train': './datasets/CelebAMask-HQ/CelebA-HQ-img',
	# 'ir_train': './datasets/CelebAMask-HQ/CelebA-HQ-img',
	'celeba_test': './datasets/CelebAMask-HQ/CelebA-HQ-img',
	'celeba_train_sketch': './datasets/CelebAMask-HQ/CelebA-HQ-edge',
	# 'celeba_test_sketch': './datasets/ir2rgb/testA',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
	'ir_test': './datasets/simslab_ir2rgb/ir',
	'ir_2_rgb_test': './datasets/simslab_ir2rgb/rgb',
	'celebA_ir_path': './datasets/CelebAMask-HQ/align_celebA_IR',
	'celebA_align_path': './datasets/CelebAMask-HQ/align_celebA',
	'celebA_subset_ir_path': './datasets/CelebAMask-HQ/align_celebA_IR_subset',
	'celebA_subset_align_path': './datasets/CelebAMask-HQ/align_celebA_subset',
	'fake_face_rgb_path': './datasets/fake_face/rgb',
	'fake_face_edge_path': './datasets/fake_face/edge',


}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
