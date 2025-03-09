import os
from PIL import Image
import scipy.io as matreader
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as Optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AdaSmoothDelta(Optim.Optimizer):
    """
    Реализация оптимизатора AdaSmoothDelta
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaSmoothDelta, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Инициализация состояния
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Обновление скользящих средних
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Вычисление шага обновления
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                denominator = exp_avg_sq.sqrt().add_(group['eps'])
                delta = exp_avg.div(denominator)

                p.data.add_(delta, alpha=-step_size)

        return loss


class CarModelsDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        data = matreader.loadmat(annotations_file)
        self.annotations = data['annotations']
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return self.annotations.shape[1]

    def __getitem__(self, idx):
        annotation = self.annotations[0, idx]
        fname = annotation['fname'][0]
        fname = str(fname)
        image_path = os.path.join(self.images_dir, fname)
        class_label = int(annotation['class'][0][0]) - 1
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_label


if __name__ == '__main__':
    cars_meta = matreader.loadmat('cars_meta.mat')
    class_names = [name[0] for name in cars_meta['class_names'][0]]

    # AlexNet ожидает входное изображение 227x227
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CarModelsDataset('cars_train_annos.mat',
                                     'cars_train',
                                     transform=transform)

    test_dataset = CarModelsDataset('cars_test_annos_withlabels_eval.mat',
                                    'cars_test',
                                    transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_adam = AlexNet(num_classes=1000).to(device)
    model_adadelta = AlexNet(num_classes=1000).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_adam = Optim.Adam(model_adam.parameters(), lr=1e-5)
    optimizer_adadelta = AdaSmoothDelta(model_adadelta.parameters(), lr=1e-5)


    def train_test(model, optimizer, train_loader, test_loader, epochs, optimizer_name):
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_dataset)
            epoch_accuracy = 100 * correct / total
            print(
                f"Эпоха {epoch + 1}, Время обучения: {round(time.time() - start_time, 2)}c., Потери: {epoch_loss:.4f}, Точность: {epoch_accuracy:.2f}%")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Тестовая точность: {test_accuracy:.2f}%")


    print("Обучение с AdaSmoothDelta")
    train_test(model_adadelta, optimizer_adadelta, train_loader, test_loader, epochs=10,
               optimizer_name="AdaSmoothDelta")
    print("\n-----------------------------------\n")
    print("Обучение с Adam")
    train_test(model_adam, optimizer_adam, train_loader, test_loader, epochs=10, optimizer_name="Adam")