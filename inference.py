import os

import torch
from unsupervised_keypoints.optimize_token import load_ldm
from unsupervised_keypoints.visualize import visualize_attn_maps

#first run scripts/demo.sh for specific text embeddings and indices

device = "cuda:0"
model_type = "runwayml/stable-diffusion-v1-5"
feature_upsample_res = 128

ldm, controllers, num_gpus = load_ldm(device, model_type, feature_upsample_res=feature_upsample_res)
current_dir = os.getcwd()
embedding_ak = torch.load(os.path.join(current_dir, "outputs_AK/embedding.pt"))
indices_ak = torch.load(os.path.join(current_dir, "outputs_AK/indices.pt"))

visualize_attn_maps(
    ldm,
    embedding_ak,
    indices_ak,
    num_tokens=500,
    layers=[0, 1, 2, 3],
    noise_level=-1,
    num_points=10,
    regressor=None,
    augment_degrees=15.0,
    augment_scale=[0.8, 1.0],
    augment_translate=[0.25, 0.25],
    dataset_loc="test_dataset/",
    save_folder="inference_ak/",
    device=device,
    dataset_name = "custom",
    controllers=controllers,
    num_gpus=num_gpus,
    max_loc_strategy="weighted_avg",
    augmentation_iterations=20,
)




