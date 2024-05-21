import cv2
import numpy as np
import os
import json
import glob
from pathlib2 import Path
from utils.DataLoader import HandPoseDataLoader

root_dir = "/mnt/d/git/dataset/handPose"

loader = HandPoseDataLoader(root_dir)

def create_output_folder(output_path):
    if not os.path.exists(Path(output_path).parent):
        os.makedirs(Path(output_path).parent)
        print(f"Created directory: {output_path}")
    else:
        print(f"Directory already exists: {output_path}")


def distort_image(image, intrinsics, distortion_coefficients):
    """
    Distort an undistorted image using the provided camera intrinsics and distortion coefficients.

    :param image: The undistorted input image
    :param intrinsics: Camera intrinsics matrix (3x3)
    :param distortion_coefficients: Distortion coefficients
    :return: The distorted output image
    """
    h, w = image.shape[:2]

    # Generate a grid of coordinates corresponding to the undistorted image
    map1, map2 = cv2.initUndistortRectifyMap(
        intrinsics, distortion_coefficients, None, intrinsics, (w, h), cv2.CV_32FC1
    )

    # Apply the inverse mapping to get the distorted image
    distorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    return distorted_image

for item in loader:
    distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion
    
    camera_intrinsics = np.array(item["intrinsics"])
    distortion_coefficients = np.array(item["distortion"])
    video_file = item["video_file"]
    # print(camera_intrinsics)
    # print(video_file)
    cap = cv2.VideoCapture(video_file)
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_path  = f"{Path(video_file).parent}/out/distorted.mp4"
    create_output_folder(output_video_path)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    print(output_video_path)

    # Undistort each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        distorted_frame = distort_image(frame, camera_intrinsics, distortion_coefficients)

        # Undistort the frame
        # undistorted_frame = cv2.undistort(frame, camera_intrinsics, distortion_coefficients)

        # Write the undistorted frame to the output video
        out.write(distorted_frame)

    # Release everything
    cap.release()
    out.release()

    print(f"Undistorted video saved at {output_video_path}")

# # Camera parameters
# camera_intrinsics = np.array([[150.0, 0.0, 255.5], [0.0, 150.0, 255.5], [0.0, 0.0, 1.0]])
# distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion

# # Open the video file
# input_video_path = '/mnt/d/git/annotations/Piano/9baa6a3d-0767-45d6-923a-fd4e85a69911/takes/iiith_piano_001_4/frame_aligned_videos/aria01_214-1.mp4'
# output_video_path = '/mnt/d/git/annotations/Piano/9baa6a3d-0767-45d6-923a-fd4e85a69911/takes/iiith_piano_001_4/frame_aligned_videos/undistorted_aria01_214-1.mp4'


# # Get video properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# # Undistort each frame
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Undistort the frame
#     undistorted_frame = cv2.undistort(frame, camera_intrinsics, distortion_coefficients)

#     # Write the undistorted frame to the output video
#     out.write(undistorted_frame)

# # Release everything
# cap.release()
# out.release()

# print(f"Undistorted video saved at {output_video_path}")
