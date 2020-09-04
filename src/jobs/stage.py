# 2-speaker
source activate torch
python train.py -cpk JointLateClusterSoftStyle4_G -tb 1 -exp 1 -speaker '["corden", "lec_cosmic"]' -model JointLateClusterSoftStyle4_G -modelKwargs '{"lambda_id": 0.1, "argmax": 1, "some_grad_flag": 1, "train_only": 1}' -note s2g_gst15 -save_dir save2/eccv_post2/tab1/s2g_gst15 -modalities '["pose/normalize", "audio/log_mel_400"]' -fs_new '[15, 15]' -input_modalities '["audio/log_mel_400"]' -output_modalities '["pose/normalize"]' -gan 1 -loss L1Loss -window_hop 5 -render 0 -batch_size 16 -num_epochs 20 -stop_thresh 3 -overfit 0 -early_stopping 0 -dev_key dev_spatialNorm -num_clusters 1 -feats '["pose", "velocity", "speed"]' -style_iters 3000 -num_iters 3000 -no_grad 0

# 4-speaker
source activate torch
python train.py -cpk JointLateClusterSoftStyle4_G -tb 1 -exp 1 -speaker '["corden", "lec_cosmic", "ytch_prof", "oliver"]' -model JointLateClusterSoftStyle4_G -modelKwargs '{"lambda_id": 0.1, "argmax": 1, "some_grad_flag": 1, "train_only": 1}' -note s2g_gst15 -save_dir save2/eccv_post2/tab1/s2g_gst15 -modalities '["pose/normalize", "audio/log_mel_400"]' -fs_new '[15, 15]' -input_modalities '["audio/log_mel_400"]' -output_modalities '["pose/normalize"]' -gan 1 -loss L1Loss -window_hop 5 -render 0 -batch_size 16 -num_epochs 20 -stop_thresh 3 -overfit 0 -early_stopping 0 -dev_key dev_spatialNorm -num_clusters 1 -feats '["pose", "velocity", "speed"]' -style_iters 3000 -num_iters 3000 -no_grad 0

# 8-speaker
source activate torch
python train.py -cpk JointLateClusterSoftStyle4_G -tb 1 -exp 1 -speaker '["corden", "lec_cosmic", "ytch_prof", "oliver", "ellen", "noah", "lec_evol", "maher"]' -model JointLateClusterSoftStyle4_G -modelKwargs '{"lambda_id": 0.1, "argmax": 1, "some_grad_flag": 1, "train_only": 1}' -note s2g_gst15 -save_dir save2/eccv_post2/tab1/s2g_gst15 -modalities '["pose/normalize", "audio/log_mel_400"]' -fs_new '[15, 15]' -input_modalities '["audio/log_mel_400"]' -output_modalities '["pose/normalize"]' -gan 1 -loss L1Loss -window_hop 5 -render 0 -batch_size 16 -num_epochs 20 -stop_thresh 3 -overfit 0 -early_stopping 0 -dev_key dev_spatialNorm -num_clusters 1 -feats '["pose", "velocity", "speed"]' -style_iters 3000 -num_iters 3000 -no_grad 0
