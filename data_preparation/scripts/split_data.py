import os
import random
import shutil
import json
import argparse


def split_dataset(annotations_dir, categories, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    combined_dir = os.path.join(annotations_dir, 'ego_pose/combined')
    train_dir = os.path.join(annotations_dir, 'ego_pose/train')
    val_dir = os.path.join(annotations_dir, 'ego_pose/val')
    test_dir = os.path.join(annotations_dir, 'ego_pose/test')

    # Ensure the split directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Annotation categories we are interested in
    for category in categories:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    # Only keep uids present in both all interested categories
    valid_uids = set(os.listdir(os.path.join(combined_dir, categories[0])))
    for category in categories[1:]:
        category_uids = set(os.listdir(os.path.join(combined_dir, category)))
        valid_uids.intersection_update(category_uids)

    total_items = len(valid_uids)
    print(f"total valid uids: {total_items}")

    valid_uids = list(valid_uids)
    random.shuffle(valid_uids)

    train_count = int(total_items * train_ratio)
    val_count = int(total_items * val_ratio)
    test_count = total_items - train_count - val_count
    print(f"train_count: {train_count}\nval_count: {val_count}\ntest_count: {test_count}")

    split_to_take_uids = {
        "train": [uid.replace('.json', '') for uid in valid_uids[:train_count]], 
        "val": [uid.replace('.json', '') for uid in valid_uids[train_count:train_count + val_count]],
        "test": [uid.replace('.json', '') for uid in valid_uids[train_count + val_count:]]
    }

    # Move the items for each category
    for category in categories:
        print(f"Splitting for category: {category}")
        category_combined_dir = os.path.join(combined_dir, category)
        print(f"category_combined_dir: {category_combined_dir}")
        
        for split, uids in split_to_take_uids.items():
            split_dir = os.path.join(annotations_dir, f'ego_pose/{split}/{category}')
            print(f"split_dir: {split_dir}")
            for uid in uids:
                if os.path.exists(os.path.join(category_combined_dir, f"{uid}.json")):
                    shutil.move(os.path.join(category_combined_dir, f"{uid}.json"), os.path.join(split_dir, f"{uid}.json"))
                    # pass
                else:
                    print(f"ERROR: UID: {uid} ({split}) not found in {category}. ")
    
    # Create the splits.json
    splits = {
        "take_uid_to_split": {},
        "split_to_take_uids": split_to_take_uids
    }

    for split, uids in split_to_take_uids.items():
        for uid in uids:
            splits["take_uid_to_split"][uid] = split

    splits_path = os.path.join(annotations_dir, "new_splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Splits saved to {splits_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Split dataset into train, val, and test sets.")
    parser.add_argument("--annotations_dir", type=str,
                        required=True, help="Path to the annotations directory")

    # parser.add_argument("--categories", type=str, nargs='+', required=True, help="List of categories to include in the split")
    categories = ['camera_pose', 'hand/annotation']

    args = parser.parse_args()

    split_dataset(args.annotations_dir, categories,
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    