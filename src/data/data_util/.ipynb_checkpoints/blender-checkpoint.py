import os
import json
import numpy as np
import imageio
import torch


trans_t = lambda t: torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).float()
        @ c2w
    )
    return c2w
def load_blender_data(
    datadir: str,
    scene_name: str,
    compare_method: str,
    train_skip: int,
    val_skip: int,
    test_skip: int,
    cam_scale_factor: float,
    white_bkgd: bool,
):
    basedir = os.path.join(datadir, scene_name)
    cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    
    splits = ["train", "val", "test"]
    metas = {}
    
    # 加载数据
    fp_train = open(os.path.join(basedir, compare_method, "transforms_train.json"), "r")
    fp_val = open(os.path.join(basedir, "transforms_val.json"), "r")
    fp_test = open(os.path.join(basedir, "transforms_test.json"), "r")
    metas["train"], metas["val"], metas["test"] = json.load(fp_train), json.load(fp_val), json.load(fp_test)
    
    images0 = []
    images1 = []
    extrinsics = []
    counts = [0]

    for s in splits:  # train/ val/ test
        meta = metas[s]
        imgs0 = []
        imgs1 = []
        poses = []

        if s == "train":
            skip = train_skip
        elif s == "val":
            skip = val_skip
        elif s == "test":
            skip = test_skip
        
        # print(f"s===>{s}, skip==========>{skip}")
        # print(meta["frames"])
        # print(meta["frames"][::skip])
        for frame in meta["frames"][::skip]:
        
            # missing_keys = [key for key in ["file_path", "haze_level_1", "haze_level_1_dark_channel"] if key not in frame]
            # if missing_keys:
            #     print(f"Skipping frame due to missing keys: {frame}. Missing keys: {missing_keys}")
            #     continue
            
            # print(frame)
            # if s=="trian":    
            # fname0 = os.path.join(basedir, "HAZE/haze_level_1")
            # fname1 = os.path.join(basedir, "HAZE/haze_level_1_dark_channel")
            # else:
            fname0 = os.path.join(basedir, frame['file_path'])
            if s =="train":
                fname1 = os.path.join(basedir, "HAZE/haze_level_2_PromptIR", frame['file_path'].split('/')[-1])
            
            #     print(fname1)
            else:
                fname1 = os.path.join(basedir, frame['file_path'])
           

            imgs0.append(imageio.imread(fname0))
            imgs1.append(imageio.imread(fname1))   
            poses.append(np.array(frame["transform_matrix"]))

        imgs0 = (np.array(imgs0) / 255.0).astype(np.float32)
        imgs1 = (np.array(imgs1) / 255.0).astype(np.float32)
        poses = np.array(poses).astype(np.float32)

        counts.append(counts[-1] + imgs0.shape[0])
        
        images0.append(imgs0)
        images1.append(imgs1)
        extrinsics.append(poses)
    


    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    
    images0 = np.concatenate(images0, 0)
    images1 = np.concatenate(images1, 0)
    extrinsics = np.concatenate(extrinsics, 0)

    # 打印extrinsics的形状以进行调试
    # print(f"extrinsics shape: {extrinsics.shape}")

    extrinsics[:, :3, 3] *= cam_scale_factor
    extrinsics = extrinsics @ cam_trans

    h, w = imgs0[0].shape[:2]
    num_frame = len(extrinsics)
    i_split += [np.arange(num_frame)]

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    intrinsics = np.array(
        [
            [[focal, 0.0, 0.5 * w], [0.0, focal, 0.5 * h], [0.0, 0.0, 1.0]]
            for _ in range(num_frame)
        ]
    )
    image_sizes = np.array([[h, w] for _ in range(num_frame)])

    # rendering poses in LOM dataset
    render_poses = torch.stack(
        [
            pose_spherical(angle, -10.0, 4.0) @ cam_trans
            for angle in np.linspace(70, 110, 30 + 1)[:-1]
        ],
        0,
    ) 

    render_poses2 = torch.stack([
            pose_spherical(110, angle2, 4.0) @ cam_trans
            for angle2 in np.linspace(-10, 10, 20 + 1)[:-1]
        ],
        0,)

    render_poses3 = torch.stack([
            pose_spherical(110*i, 10, 4.0*i) @ cam_trans
            for i in np.linspace(1, 0.8, 20 + 1)[:-1]
        ],
        0,)

    render_poses = torch.cat((render_poses,render_poses2,render_poses3), 0)

    render_poses[:, :3, 3] *= cam_scale_factor
    near = 2.0
    far = 6.0

    if white_bkgd:
        images0 = images0[..., :3] * images0[..., -1:] + (1.0 - images0[..., -1:])
        images1 = images1[..., :3] * images1[..., -1:] + (1.0 - images1[..., -1:])
    else:
        images0 = images0[..., :3]
        images1 = images1[..., :3]
        
    return (
        images0,
        images1,
        intrinsics,
        extrinsics,
        image_sizes,
        near,
        far,
        (-1, -1),
        i_split,
        render_poses,
    )
