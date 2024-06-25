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


# test-para
# f1_macro: 0.9393333333333332, f1_weighted: 0.9368421052631579,
# accuracy: 93.6842105263158, average_test_loss: 0.20148823354393244
# Class: Lung_Opacity, F1 score: 0.888
# Class: Normal, F1 score: 0.94
# Class: Viral Pneumonia, F1 score: 0.99
para = [0.001, 0.9, 25]
if __name__ == '__main__':
    # 加载预训练的EfficientNet模型
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(train.class_names))
    model = model.to(train.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=para[0], momentum=para[1])

    model = train.train_model(model, criterion, optimizer, train.dataloaders, train.dataset_sizes, train.device, num_epochs=para[2])

    # 保存模型
    torch.save(model.state_dict(), 'lung_xray_efficientnet.pth')

    print('训练完成并保存模型')
