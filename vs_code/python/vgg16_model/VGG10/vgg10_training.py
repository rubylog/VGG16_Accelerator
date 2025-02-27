import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ✅ CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 현재 사용 중인 디바이스 출력

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # RGB mean, var
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)  # ✅ num_workers 최적화

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# Model, loss function, and optimizer
model = VGG10(num_classes=10).to(device)  # ✅ 모델을 GPU로 이동
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 함수 (GPU 적용)
def train_model(num_epochs=100):
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # 학습 모드 설정
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)  # ✅ 데이터를 GPU로 이동

            optimizer.zero_grad()
            outputs = model(inputs)  # (batch size, 10)
            loss = criterion(outputs, labels)  # softmax & NLL loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')
    print('Training complete')

# Train the model
train_model()

# ✅ 테스트 정확도 평가 (GPU 적용)
correct = 0
total = 0
model.eval()  # 평가 모드 설정
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)  # ✅ 데이터를 GPU로 이동
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)  # batch size
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')  # 최종 정확도 출력
