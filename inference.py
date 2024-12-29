import os

import torch
from unsupervised_keypoints.optimize_token import load_ldm
from unsupervised_keypoints.visualize import visualize_attn_maps

device = "cuda:0"
model_type = "runwayml/stable-diffusion-v1-5"
feature_upsample_res = 128

ldm, controllers, num_gpus = load_ldm(device, model_type, feature_upsample_res=feature_upsample_res)
current_dir = os.getcwd()
embedding_celeba = torch.load(os.path.join(current_dir, "celeba/embedding.pt"))
indices_celeba = torch.load(os.path.join(current_dir, "celeba/indices.pt"))
embedding_human = torch.load(os.path.join(current_dir, "unaligned_human3.6m/embedding.pt"))
indices_human = torch.load(os.path.join(current_dir, "unaligned_human3.6m/indices.pt"))

# visualize_attn_maps(
#     ldm,
#     embedding_celeba,
#     indices_celeba,
#     num_tokens=500,
#     layers=[0, 1, 2, 3],
#     noise_level=-1,
#     num_points=10,
#     regressor=None,
#     augment_degrees=15.0,
#     augment_scale=[0.8, 1.0],
#     augment_translate=[0.25, 0.25],
#     dataset_loc="MPII/images/",
#     save_folder="inference_celeba/",
#     device=device,
#     dataset_name = "custom",
#     controllers=controllers,
#     num_gpus=num_gpus,
#     max_loc_strategy="argmax",
#     augmentation_iterations=10,
# )

visualize_attn_maps(
    ldm,
    embedding_human,
    indices_human,
    num_tokens=500,
    layers=[0, 1, 2, 3],
    noise_level=-1,
    num_points=16,
    regressor=None,
    augment_degrees=15.0,
    augment_scale=[0.8, 1.0],
    augment_translate=[0.25, 0.25],
    dataset_loc="MPII/images/",
    save_folder="inference_human/",
    device=device,
    dataset_name = "custom",
    controllers=controllers,
    num_gpus=num_gpus,
    max_loc_strategy="weighted_avg",
    augmentation_iterations=10,
)




