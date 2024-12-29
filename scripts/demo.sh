export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python -m unsupervised_keypoints.main \
--dataset_loc ./MPII/images/ \
--dataset_name custom \
--num_steps 250 \
--max_num_points 500 \
--evaluation_method orientation_invariant \
--save_folder ./outputs_MPII