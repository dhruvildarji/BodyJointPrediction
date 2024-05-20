import os
import json
import subprocess
import wexpect

# Define the directory containing the JSON files and the output directory for the videos
json_dir = "D:/git/annotations/annotations/ego_pose/train/camera_pose"


# Function to extract uid and check if the take_name contains "basketball"
def extract_uid_if_basketball(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        if 'basketball' in data['metadata']['take_name']:
            return data['metadata']['take_uid']
    return None

# Function to run the egoexo command with the extracted uid, defaulting the confirmation to "yes"
def download_ego_video(uid, output_dir, c):
    # command = f"echo y | egoexo -o {output_dir} --uids {uid} --parts takes --views ego"
    command = f"egoexo -o {output_dir} --uids {uid} --parts takes --views ego"
    print(command)
    print(f"download video {c}")    
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.stdin.write(b'y\n')
    process.stdin.flush()
    stdout, stderr = process.communicate()
    
    print(stdout.decode())
    if stderr:
        print(stderr.decode())

c = 0
# Iterate through each JSON file in the specified directory
for json_filename in os.listdir(json_dir):
    if json_filename.endswith('.json'):
        json_file_path = os.path.join(json_dir, json_filename)
        
        # Extract uid from the current JSON file
        uid = extract_uid_if_basketball(json_file_path)

        output_dir = f"D:/git/annotations/basketball/{uid}"
        # Download the ego video using the extracted uid
        if uid != None:
            download_ego_video(uid, output_dir, c)
            c = c + 1

print(f"All videos have been downloaded. total videos downloaded{c}")

