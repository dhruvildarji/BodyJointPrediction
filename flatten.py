import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a function to read the frames from a video
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Define a function to write frames to a video
def write_video_frames(frames, output_path, fps=20.0):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

# Define a function to flatten the image
def flatten_image(frame, intrinsics, extrinsics):
    h, w = frame.shape[:2]
    intrinsic_matrix = np.array(intrinsics)
    
    # Convert extrinsics to 3x3 (rotation) and translation vector
    rotation_matrix = np.array([extrinsics[i][:3] for i in range(3)])
    translation_vector = np.array([extrinsics[i][3] for i in range(3)]).reshape((3, 1))

    # Combine rotation and translation into a single 3x4 matrix
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))

    # Construct the homography matrix (3x3) from the intrinsic and extrinsic parameters
    homography_matrix = intrinsic_matrix @ np.hstack((rotation_matrix, translation_vector))[:, :3]

    # Normalize the homography matrix
    homography_matrix /= homography_matrix[2, 2]

    # Print matrices for debugging
    print("Intrinsic Matrix:\n", intrinsic_matrix)
    print("Rotation Matrix:\n", rotation_matrix)
    print("Translation Vector:\n", translation_vector)
    print("Homography Matrix:\n", homography_matrix)

    # Apply the transformation
    flattened_frame = cv2.warpPerspective(frame, homography_matrix, (w, h))

    return flattened_frame

# Camera parameters
intrinsics = [[150.0, 0.0, 255.5], [0.0, 150.0, 255.5], [0.0, 0.0, 1.0]]
extrinsics = [[0.25332655304165963, -0.16657374284029, -0.9529317117262677, 0.984832189787714],
              [0.38694833037292553, 0.920275535674013, -0.05799938575489, 0.6950308525824895],
              [0.8866209159267855, -0.3540425494132491, 0.2975856609614858, 2.385611958598334]]

# Paths
input_video_path = "/mnt/d/git/annotations/basketball/0a6f112f-6cd8-4c53-adda-0d7862804b87/takes/unc_basketball_02-24-23_01_27/frame_aligned_videos/aria01_214-1.mp4"
output_video_path = "/mnt/d/git/annotations/basketball/0a6f112f-6cd8-4c53-adda-0d7862804b87/takes/unc_basketball_02-24-23_01_27/flattened_aria01_214-1.mp4"

# Read frames from the original video
original_frames = read_video_frames(input_video_path)

# Flatten each frame and visualize
flattened_frames = []
for i, frame in enumerate(original_frames):
    flattened_frame = flatten_image(frame, intrinsics, extrinsics)
    flattened_frames.append(flattened_frame)
    
    # Visualize the original and flattened frame
    if i < 5:  # Show the first 5 frames for inspection
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Original Frame')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(flattened_frame, cv2.COLOR_BGR2RGB))
        plt.title('Flattened Frame')
        plt.show()

# Write the flattened frames to a new video
write_video_frames(flattened_frames, output_video_path)
