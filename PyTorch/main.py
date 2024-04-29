import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1_input_size = 32 * 8 * 8
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define transformations for train and test datasets
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Load train dataset
train_dataset = ImageFolder(root=r"C:\Users\Tejas Tyagi\Downloads\archive (2)\kagglecatsanddogs_3367a\PetImages", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Initialize the model and move it to the GPU
net = SimpleCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop with progress bar
for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}', unit='batch', leave=False) as pbar:
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.update(1)
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)} ,Accuracy: {(100 * correct / total):.2f}%')
print('Finished Training')

# Save the trained model
torch.save(net.state_dict(), 'simple_cnn.pth')

# Load the trained model
net = SimpleCNN().to(device)
net.load_state_dict(torch.load('simple_cnn.pth'))

# Evaluation
net.eval()

# Load and preprocess test image
test_transform = transforms.Compose([
    transforms.Resize((720, 720)),
    transforms.ToTensor(),
])
test_image = test_transform(Image.open(r"C:\Users\Tejas Tyagi\Downloads\archive (2)\kagglecatsanddogs_3367a\PetImages\Cat\9884.jpg")).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = net(test_image)
    _, predicted_class = torch.max(output, 1)

class_names = train_dataset.classes
print(f'The model predicts the image as: {class_names[predicted_class.item()]}')
