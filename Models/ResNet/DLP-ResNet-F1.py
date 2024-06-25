import torch
import torch.nn as nn
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
import Models.TestModel as test


if __name__ == '__main__':
    # 加载模型
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(test.class_names))
    model.load_state_dict(torch.load('lung_xray_resnet18.pth'))
    model = model.to(test.device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 运行测试循环
    f1_macro, f1_weighted, accuracy, average_test_loss, class_f1_scores = test.test_loop(test.dataloaders, model,
                                                                                         criterion)

    print(
        f"f1_macro: {f1_macro}, f1_weighted: {f1_weighted}, accuracy: {accuracy}, average_test_loss: {average_test_loss}")
    for class_name, class_f1 in class_f1_scores.items():
        print(f"Class: {class_name}, F1 score: {class_f1}")
