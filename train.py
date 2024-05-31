import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import HandPoseDataLoader, HandPoseDataset
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet18(pretrained=True)  # Using ResNet-18 for this example
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers (avgpool and fc)
        self.fc1 = nn.Linear(resnet.fc.in_features, 512)
        self.fc2 = nn.Linear(512, 21 * 3)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 21, 3)

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}...")

    for epoch in range(num_epochs):
        model.train()  # Ensure the model is in training mode
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
    hand_pose_loader = HandPoseDataLoader("/Users/rdhara/Downloads/cs231n/git/cs231project/cs231_dataset")

    print("Creating dataset...")
    dataset = HandPoseDataset(hand_pose_loader)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    print("Initializing model...")
    model = ResNetBackbone()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

if __name__ == "__main__":
    main()
