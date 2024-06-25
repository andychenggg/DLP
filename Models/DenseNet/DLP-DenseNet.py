import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import sys
import os
import types
sys.modules["__main__"].__spec__ = types.SimpleNamespace()
# 获取当前脚本的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
# 添加项目根目录到系统路径
sys.path.append(project_root)
import Models.TrainModel as train

# test-para:
# f1_macro: 0.9417393308334768, f1_weighted: 0.9407416672685484,
# accuracy: 94.10526315789474, average_test_loss: 0.342020068866744
# Class: Lung_Opacity, F1 score: 0.9105691056910571
# Class: Normal, F1 score: 0.9437751004016064
# Class: Viral Pneumonia, F1 score: 0.970873786407767
para = [0.001, 0.9, 25]
if __name__ == '__main__':
    print(f'Using device: {train.device}')
    # 加载预训练的ResNet模型
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(train.class_names))
    model = model.to(train.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=para[0], momentum=para[1])

    model = train.train_model(
        model, criterion, optimizer,
        train.dataloaders, train.dataset_sizes,
        train.device, num_epochs=para[2]
    )

    # 保存模型
    torch.save(model.state_dict(), 'lung_xray_densenet121.pth')

    print('训练完成并保存模型')
