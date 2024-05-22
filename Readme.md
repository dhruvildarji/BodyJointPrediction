# HandPose Detection

## Need to Fill Details

# HandPose Data Loader

## Overview

The `HandPoseDataLoader` class is designed to load and preprocess hand pose data from the EgoExo meta dataset for neural network training. This class handles the loading of JSON annotations, video files, and organizes the data into a format that can be easily used with PyTorch's `DataLoader`.

## HandPoseDataLoader Class

### Initialization

The class is initialized with the root directory of the dataset and the type of data (e.g., "train" or "test"). It supports loading data from a pickle file to speed up subsequent data loading operations.

```python
def __init__(self, root_dir, data_type="train", use_pickle=False):
```

root_dir: The root directory where the dataset is stored.
data_type: The type of data to load (default is "train").
use_pickle: If set to True, it will load data from a pickle file if available.
Methods
find_json_files()
This method finds all JSON files in the specified directory for annotations.

find_videos_with_214()
This method searches for video files with "214" in their names within the dataset directory.

get_video_dir()
This method lists all video directories in the root video directory.

load_hand_data()
This method loads hand pose annotations from JSON files and integrates them into the main data structure.

get_int_from_dict(dictionary_key, target_string)
A helper method to retrieve an integer key from a dictionary based on a target string value.

init_videos()
This method initializes video capture objects for each video file and extracts frames corresponding to the annotated hand poses.

load_data()
This method loads the main JSON annotations into the self.data dictionary.

get_frames(file_name)
This method retrieves frames for a specific file name.

load_videos()
This method populates the self.video_files dictionary with video file paths.

get_intrinsics(file_name)
This method retrieves the camera intrinsics for a specific file name.

get_distortion(file_name)
This method retrieves the camera distortion coefficients for a specific file name.

get_hand_poses(file_name)
This method retrieves the hand pose annotations for a specific file name.

release()
This method releases all video capture objects.

__iter__()
Initializes the iteration.

__next__()
Retrieves the next item in the dataset for iteration.

Detailed Steps for Training a Neural Network with HandPoseDataLoader
Initialize DataLoader:

Ensure that the dataset root directory is correctly set.
Instantiate HandPoseDataLoader with the desired parameters.
Create Custom Dataset:

Use the HandPoseDataset class to wrap the data loader.
PyTorch DataLoader:

Create a PyTorch DataLoader instance with the custom dataset.
Set the desired batch size, shuffling, and number of workers.
Training Loop:

Iterate over the DataLoader to fetch batches.
Pass the data batches to the neural network for training.
By following these steps, you can effectively use the HandPoseDataLoader and HandPoseDataset to train a neural network for estimating hand poses. Ensure to handle any specific requirements of your neural network model, such as input shapes and preprocessing steps.