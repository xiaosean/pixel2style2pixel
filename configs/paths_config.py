dataset_paths = {
	'celeba_train': './datasets/CelebAMask-HQ/CelebA-HQ-img',
	# 'ir_train': './datasets/CelebAMask-HQ/CelebA-HQ-img',
	'celeba_test': './datasets/CelebAMask-HQ/CelebA-HQ-img',
	'celeba_train_sketch': './datasets/ir2rgb/trainA',
	'celeba_test_sketch': './datasets/ir2rgb/testA',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
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
