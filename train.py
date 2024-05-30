tr  import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from BodyJointPrediction.dataloader import HandPoseDataLoader
import torch.nn.functional as F

class HandPoseDataset(Dataset):
    def __init__(self, hand_pose_loader):
        self.data = []s
        for item in hand_pose_loader:
            frames = item['frames']
            hand_poses = item['hand_pose']
            if frames is not None and hand_poses is not None:
                for frame, hand_pose in zip(frames, hand_poses.values()):
                    self.data.append((frame, hand_pose))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame, hand_pose = self.data[idx]
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize and permute for channels
        hand_pose = torch.tensor(hand_pose, dtype=torch.float32)
        return frame, hand_pose

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 21 * 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 21, 3)

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}...")

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    print("Finished Training")

def main():
    print("Initializing data loader...")
    hand_pose_loader = HandPoseDataLoader("/Users/rdhara/Downloads/ego-exo4d-egopose/handpose/cs231project/dataset")

    print("Creating dataset...")
    dataset = HandPoseDataset(hand_pose_loader)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    print("Initializing model...")
    model = SimpleCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

if __name__ == "__main__":
    main()