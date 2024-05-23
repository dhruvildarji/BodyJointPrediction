import numpy as np


def get_ego_aria_cam_name(take):
    ego_cam_names = [
        x["cam_id"]
        for x in take["capture"]["cameras"]
        if str(x["is_ego"]).lower() == "true"
    ]
    assert len(ego_cam_names) > 0, "No ego cameras found!"
    if len(ego_cam_names) > 1:
        ego_cam_names = [
            cam for cam in ego_cam_names if cam in take["frame_aligned_videos"].keys()
        ]
        assert len(ego_cam_names) > 0, "No frame-aligned ego cameras found!"
        if len(ego_cam_names) > 1:
            ego_cam_names_filtered = [
                cam for cam in ego_cam_names if "aria" in cam.lower()
            ]
            if len(ego_cam_names_filtered) == 1:
                ego_cam_names = ego_cam_names_filtered
        assert (
            len(ego_cam_names) == 1
        ), f"Found too many ({len(ego_cam_names)}) ego cameras: {ego_cam_names}"
    ego_cam_names = ego_cam_names[0]
    return ego_cam_names


def world_to_cam(kpts, extri):
    """
    Transform 3D world kpts to camera coordinate system
    Input:
        kpts: (N,3)
        extri: (3,4) [R|t]
    Output:
        new_kpts: (N,3)
    """
    new_kpts = kpts.copy()
    new_kpts = np.append(new_kpts, np.ones(
        (new_kpts.shape[0], 1)), axis=1).T  # (4,N)
    new_kpts = (extri @ new_kpts).T  # (N,3)
    return new_kpts


def cam_to_img(kpts, intri):
    """
    Project points in camera coordinate system to image plane
    Input:
        kpts: (N,3)
    Output:
        new_kpts: (N,2)
    """
    new_kpts = kpts.copy()
    new_kpts = intri @ new_kpts.T  # (3,N)
    new_kpts = new_kpts / new_kpts[2, :]
    new_kpts = new_kpts[:2, :].T
    return new_kpts
