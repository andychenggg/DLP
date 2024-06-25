import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import sys
import types
import os
sys.modules["__main__"].__spec__ = types.SimpleNamespace()
# 获取当前脚本的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
# 添加项目根目录到系统路径
sys.path.append(project_root)
import Models.TrainModel as train


# test=para
# f1_macro: 0.9493490237975531, f1_weighted: 0.9487088526368712,
# accuracy: 94.94736842105263, average_test_loss: 0.26309212507718865
# Class: Lung_Opacity, F1 score: 0.9145299145299145
# Class: Normal, F1 score: 0.953125
# Class: Viral Pneumonia, F1 score: 0.9803921568627451
para = [0.001, 0.9, 25]
if __name__ == '__main__':
    print(f'Using device: {train.device}')
    # 加载预训练的ResNet模型
    weights = models.ResNet152_Weights.DEFAULT
    model = models.resnet152(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train.class_names))
    model = model.to(train.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=para[0], momentum=para[1])

    model = train.train_model(model, criterion, optimizer, train.dataloaders, train.dataset_sizes, train.device, num_epochs=para[2])

    # 保存模型
    torch.save(model.state_dict(), 'lung_xray_resnet18.pth')

    print('训练完成并保存模型')
