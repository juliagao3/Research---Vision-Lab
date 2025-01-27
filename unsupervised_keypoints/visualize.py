# load the dataset
import os
import torch
from torchgen.dest.register_dispatch_key import gen_resize_out_helper

import numpy as np
from glob import glob
from tqdm import tqdm
from unsupervised_keypoints import ptp_utils
import torch.nn.functional as F
from datasets.celeba import CelebA
from datasets import custom_images
from datasets import cub
from datasets import cub_parts
from datasets import taichi
from datasets import human36m
from datasets import unaligned_human36m
from datasets import deepfashion
from unsupervised_keypoints.eval import run_image_with_context_augmented
from unsupervised_keypoints.eval import pixel_from_weighted_avg, find_max_pixel, mask_radius, find_k_max_pixels

from unsupervised_keypoints.invertable_transform import RandomAffineWithInverse

import matplotlib.pyplot as plt
import torchvision

def save_img(map, img, name):
    # save with matplotlib
    # map is shape [32, 32]
    import matplotlib.pyplot as plt

    plt.imshow(map.cpu().detach().numpy())
    plt.title(f"max: {torch.max(map).cpu().detach().numpy()}")
    plt.savefig(f"outputs/{name}_map.png")
    plt.close()
    # save original image current with shape [3, 512, 512]
    plt.imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
    plt.savefig(f"outputs/{name}_img.png")
    plt.close()


def save_grid(maps, imgs, name, img_size=(512, 512), dpi=50, quality=85):
    """
    There are 10 maps of shape [32, 32]
    There are 10 imgs of shape [3, 512, 512]
    Saves as a single image with matplotlib with 2 rows and 10 columns
    Updated to have smaller borders between images and the edge.
    DPI is reduced to decrease file size.
    JPEG quality can be adjusted to trade off quality for file size.
    """

    # Calculate figure size to maintain aspect ratio
    fig_width = img_size[1] * 4  # total width for 10 images side by side
    fig_height = img_size[0] * 2  # total height for 2 images on top of each other
    fig_size = (fig_width / 100, fig_height / 100)  # scale down to a manageable figure size

    fig, axs = plt.subplots(2, 4, figsize=fig_size, gridspec_kw={'wspace':0.05, 'hspace':0.05})

    for i in range(4):
        axs[0, i].imshow(imgs[i].numpy().transpose(1, 2, 0))
        axs[1, i].imshow(imgs[i].numpy().transpose(1, 2, 0))
        normalized_map = maps[i] - torch.min(maps[i])
        normalized_map = normalized_map / torch.max(normalized_map)
        axs[1, i].imshow(normalized_map, alpha=0.7)

    # Remove axis and adjust subplot parameters
    for ax in axs.flatten():
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save as JPEG with reduced DPI and specified quality
    plt.savefig(name, format='jpg', bbox_inches='tight', pad_inches=0, dpi=dpi, pil_kwargs={'quality': quality})

    plt.close()


def plot_point_single(img, points, name):
    """
    Displays corresponding points on the image with white outline around plotted numbers.
    The numbers themselves retain their original color.
    points shape is [num_people, num_points, 2]
    """
    num_people, num_points, _ = points.shape

    # Get the default color cycle from Matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img.numpy().transpose(1, 2, 0))

    for i in range(num_people):
        for j in range(num_points):
            # Choose color based on j, cycling through the default color cycle
            color = colors[j % len(colors)]
            x, y = points[i, j, 1] * 512, points[i, j, 0] * 512
            # Plot the original color on top
            ax.scatter(x, y, color=color, marker=f"${j}$", s=300)

    ax.axis("off")  # Remove axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove border

    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

def plot_point_correspondences(imgs, points, name, height = 11, width = 9, ground_truth=False):
    """
    Displays corresponding points per image
    len(imgs) = num_images
    points shape is [num_images, num_points, 2]
    """

    num_images, num_points, _ = points.shape

    fig, axs = plt.subplots(height, width, figsize=(2 * width, 2 * height))
    axs = axs.ravel()  # Flatten the 2D array of axes to easily iterate over it

    for i in range(height*width):
        axs[i].imshow(imgs[i].numpy().transpose(1, 2, 0))

        for j in range(num_points):
            # plot the points each as a different type of marker
            x = points[i, j, 1] * 512
            y = points[i, j, 0] * 512

            if ground_truth:
                if 0 <= x <= 512 and 0 <= y <= 512:
                    axs[i].scatter(
                        y, x, marker=f"${j}$"
                    )
            else:
                axs[i].scatter(
                    x, y, marker=f"${j}$"
                )

    # remove axis and handle any unused subplots
    for i, ax in enumerate(axs):
        if i >= num_images:
            ax.axis("off")  # Hide unused subplots
        else:
            ax.axis("off")  # Remove axis from used subplots

    # Adjust subplot parameters to reduce space between images and border space
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    # increase the resolution of the plot
    plt.savefig(name, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

@torch.no_grad()
def visualize_attn_maps(
    ldm,
    context,
    indices,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=32,
    layers=[0, 1, 2, 3, 4, 5],
    lr=5e-3,
    noise_level=-1,
    num_tokens=1000,
    num_points=30,
    num_images=100,
    regressor=None,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augmentation_iterations=20,
    dataset_loc="~",
    save_folder="outputs",
    visualize=False,
    dataset_name = "celeba_aligned",
    controllers=None,
    num_gpus=1,
    max_loc_strategy="argmax",
    height = 11,
    width = 9,
    validation = False,
):
    if dataset_name == "celeba_aligned":
        dataset = CelebA(split="test", dataset_loc=dataset_loc)
    elif dataset_name == "celeba_wild":
        dataset = CelebA(split="test", dataset_loc=dataset_loc, align = False)
    elif dataset_name == "cub_aligned":
        dataset = cub.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "cub_001":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test", single_class=1)
    elif dataset_name == "cub_002":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test", single_class=2)
    elif dataset_name == "cub_003":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test", single_class=3)
    elif dataset_name == "cub_all":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test")
    elif dataset_name == "taichi":
        dataset = taichi.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "human3.6m":
        dataset = human36m.TestSet(data_root=dataset_loc, validation=validation)
    elif dataset_name == "unaligned_human3.6m":
        dataset = unaligned_human36m.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "deepfashion":
        dataset = deepfashion.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "custom":
        dataset = custom_images.CustomDataset(data_root=dataset_loc, image_size=512)
    else:
        raise NotImplementedError

    imgs = []
    maps = []
    gt_kpts = []
    
    # random permute the dataset
    randperm = torch.randperm(len(dataset))
    
    for i in tqdm(range(height * width)):
        batch = dataset[randperm[i%len(dataset)].item()]

        img = batch["img"]

        _gt_kpts = batch["kpts"] 
        gt_kpts.append(_gt_kpts)
        imgs.append(img.cpu())

        map = run_image_with_context_augmented(
            ldm,
            img,
            context,
            indices.cpu(),
            device=device,
            from_where=from_where,
            layers=layers,
            noise_level=noise_level,
            augment_degrees=augment_degrees,
            augment_scale=augment_scale,
            augment_translate=augment_translate,
            augmentation_iterations=augmentation_iterations,
            visualize=(i==0),
            controllers=controllers,
            num_gpus=num_gpus,
            save_folder=save_folder,
        )

        maps.append(map.cpu())
    maps = torch.stack(maps)
    gt_kpts = torch.stack(gt_kpts)


    gt_kpts = gt_kpts / torch.tensor([640, 360])
    plot_point_correspondences(
        imgs, gt_kpts, os.path.join(save_folder, "gt_keypoints.pdf"), height, width, ground_truth=True
    )

    if max_loc_strategy == "argmax":
        points = find_max_pixel(maps.view(height * width * num_points, 512, 512)) / 512.0
    else:
        points = pixel_from_weighted_avg(maps.view(height * width * num_points, 512, 512)) / 512.0
    points = points.reshape(height * width, num_points, 2)

    plot_point_correspondences(
        imgs, points.cpu(), os.path.join(save_folder, "unsupervised_keypoints.pdf"), height, width
    )

    for i in range(num_points):
        save_grid(
            maps[:, i].cpu(), imgs, os.path.join(save_folder, f"keypoint_{i:03d}.png")
        )

    if regressor is not None:
        points = points.to(device)
        est_points = ((points.view(num_images, -1)-0.5) @ regressor)+0.5

        plot_point_correspondences(
            imgs,
            est_points.view(num_images, -1, 2).cpu(),
            os.path.join(save_folder, "estimated_keypoints.pdf"),
            height,
            width,
        )

        plot_point_correspondences(
            imgs, gt_kpts, os.path.join(save_folder, "gt_keypoints.pdf"), height, width
        )
        
        
@torch.no_grad()
def create_vid(
    ldm,
    contexts,
    indices,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    noise_level=-1,
    num_points=30,
    num_images=100,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augment_shear=(0.0, 0.0),
    augmentation_iterations=20,
    dataset_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/",
    save_folder="outputs",
    controllers=None,
    num_gpus=1,
    max_loc_strategy="argmax",
    dataset_name = "celeba_aligned",
    validation=False,
    max_num_frames = 1_000,
):
    if dataset_name == "celeba_aligned":
        dataset = CelebA(split="test", dataset_loc=dataset_loc)
    elif dataset_name == "celeba_wild":
        dataset = CelebA(split="test", dataset_loc=dataset_loc, align = False)
    elif dataset_name == "cub_aligned":
        dataset = cub.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "cub_001":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test", single_class=1)
    elif dataset_name == "cub_002":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test", single_class=2)
    elif dataset_name == "cub_003":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test", single_class=3)
    elif dataset_name == "cub_all":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test")
    elif dataset_name == "taichi":
        dataset = taichi.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "human3.6m":
        dataset = human36m.TestSet(data_root=dataset_loc, validation=validation)
    elif dataset_name == "unaligned_human3.6m":
        dataset = unaligned_human36m.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "deepfashion":
        dataset = deepfashion.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "custom":
        dataset = custom_images.CustomDataset(data_root=dataset_loc, image_size=512)
    else:
        raise NotImplementedError
    
    # make a random permutation of the dataset
    # randperm = torch.randperm(len(dataset))
    randperm = torch.arange(len(dataset))
    # randperm = torch.arange(len(dataset))
    
    keypoints = []
    saved_maps = []
    
    for i in tqdm(range(len(randperm))):
        
        # interpolate exponentially between 0 and 500 over len(randperm)
        
        # index = min(499, int((i*0.05)**2.3))
        
        # create_progress_bar(index, len(randperm), f'satya/progress_{i:04d}.png')
        
        # print(index, i, end = "\r")
        
        # index = min(499, i*2)
        
        
        
        batch = dataset[randperm[i].item()]

        img = batch["img"]

        maps = []
        for j in range(len(contexts)):
            map = run_image_with_context_augmented(
                ldm,
                img,
                contexts[j],
                indices.cpu(),
                device=device,
                from_where=from_where,
                layers=layers,
                noise_level=noise_level,
                augment_degrees=augment_degrees,
                augment_scale=augment_scale,
                augment_translate=augment_translate,
                augmentation_iterations=augmentation_iterations,
                controllers=controllers,
                num_gpus=num_gpus,
                save_folder=save_folder,
                upscale_size=128,
            )
            maps.append(map)
            
        maps = torch.stack(maps)
        map = torch.mean(maps, dim=0)
        
        saved_maps.append(map.cpu())
        
        # plot_map_single(
        #     img, map, os.path.join(save_folder, f"unsupervised_keypoints_{i:04d}.png")
        # )
        
        point = find_max_pixel(map) / 512.0
        
        # plot_point_single(
        #     img, point.unsqueeze(0).cpu(), os.path.join(save_folder, f"unsupervised_keypoints_{i:04d}.png")
        # )
        
        keypoints.append(point)
        
    keypoints = torch.stack(keypoints)
    # save as a json file
    torch.save(keypoints, os.path.join(save_folder, "keypoints.pt"))
    torch.save(saved_maps, os.path.join(save_folder, "saved_maps.pt"))
    
    # import json
    # with open(os.path.join(save_folder, "keypoints.json"), "w") as f:
    #     json.dump(keypoints.tolist(), f)

# def _calc_distances(preds, targets, mask, normalize):
#     print("preds", preds)
#     print("targets", targets)
#     N, K, D = preds.shape
#     #_mask[(normalize == 0).sum(1).nonzero(as_tuple=True)[0], :] = False
#     distances = torch.full((N, K), -1, dtype=torch.float32)
#     #normalize[normalize <= 0] = 1e6
#
#     for i in range(N):
#         for j in range(K):
#             if mask[i, j] == 1:
#                 distance = (preds[i, j] - targets[i, j]) / normalize[i, :]
#                 distances[i, j] = torch.norm(distance)
#
#     print("calc distances", distances)
#     return distances.T

# def _distance_acc(distances, threshold=0.05):
#     print("distances", distances)
#     distance_valid = distances != -1
#     num_distance_valid = distance_valid.sum()
#     if num_distance_valid > 0:
#         print((distances[distance_valid] < threshold).sum())
#         return (distances[distance_valid] < threshold).sum() / num_distance_valid
#     return -1
#
# def keypoint_pck_accuracy(preds, targets, mask, threshold, normalize):
#     distances = _calc_distances(preds, targets, mask, normalize)
#
#     acc = np.array([_distance_acc(d, threshold) for d in distances])
#     print("acc", acc)
#     valid_acc = []
#     for a in acc:
#         if a <= 0:
#             continue
#         else:
#             valid_acc.append(a)
#     print(valid_acc)
#     count = len(valid_acc)
#     avg_acc = np.mean(valid_acc) if count > 0 else 0
#     return acc, avg_acc, count
#
# def evaluate(
#         ldm,
#         context,
#         indices,
#         device="cuda",
#         from_where=["down_cross", "mid_cross", "up_cross"],
#         upsample_res=32,
#         layers=[0, 1, 2, 3, 4, 5],
#         lr=5e-3,
#         noise_level=-1,
#         num_tokens=1000,
#         num_points=30,
#         num_images=100,
#         regressor=None,
#         augment_degrees=30,
#         augment_scale=(0.9, 1.1),
#         augment_translate=(0.1, 0.1),
#         augmentation_iterations=20,
#         dataset_loc="~",
#         save_folder="outputs",
#         visualize=False,
#         dataset_name="celeba_aligned",
#         controllers=None,
#         num_gpus=1,
#         max_loc_strategy="argmax",
#         height=11,
#         width=9,
#         validation=False,
# ):
#     dataset = custom_images.CustomDataset(data_root=dataset_loc, image_size=512)
#
#     imgs = []
#     maps = []
#     gt_kpts = []
#     masks = []
#
#     # random permute the dataset
#     randperm = torch.randperm(len(dataset))
#
#     for i in tqdm(range(height * width)):
#         batch = dataset[randperm[i % len(dataset)].item()]
#
#         img = batch["img"]
#
#         _gt_kpts = batch["kpts"]
#         gt_kpts.append(_gt_kpts)
#         imgs.append(img.cpu())
#
#         mask = batch["mask"]
#         masks.append(mask)
#
#         map = run_image_with_context_augmented(
#             ldm,
#             img,
#             context,
#             indices.cpu(),
#             device=device,
#             from_where=from_where,
#             layers=layers,
#             noise_level=noise_level,
#             augment_degrees=augment_degrees,
#             augment_scale=augment_scale,
#             augment_translate=augment_translate,
#             augmentation_iterations=augmentation_iterations,
#             visualize=(i == 0),
#             controllers=controllers,
#             num_gpus=num_gpus,
#             save_folder=save_folder,
#         )
#
#         maps.append(map.cpu())
#     maps = torch.stack(maps)
#     gt_kpts = torch.stack(gt_kpts)
#     masks = torch.stack(masks)
#
#     if max_loc_strategy == "argmax":
#         points = find_max_pixel(maps.view(height * width * num_points, 512, 512))
#     else:
#         points = pixel_from_weighted_avg(maps.view(height * width * num_points, 512, 512))
#     points = points.reshape(height * width, num_points, 2)
#
#     gt_kpts = gt_kpts[:, :num_points, :]
#     masks = masks[:, :num_points]
#
#     print(maps.shape)
#     N, K, H, W = maps.shape
#     if K == 0:
#         return None, 0, 0
#     normalize = torch.tile(torch.tensor([[H, W]]), (N, 1))
#     print("normalize", normalize)
#
#     acc, avg_acc, count = keypoint_pck_accuracy(points, gt_kpts, masks, 0.05, normalize)
#     print(acc, avg_acc, count)