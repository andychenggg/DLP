import sys
import types
import os
import torch
import torch.nn as nn
from torchvision import models
import Models.TestModel as test
import optuna
import torch.optim as optim
sys.modules["__main__"].__spec__ = types.SimpleNamespace()
# 获取当前脚本的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
# 添加项目根目录到系统路径
sys.path.append(project_root)
import Models.TrainModel as train


def objective(trial):
    # 超参数建议
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.9)
    num_epochs = trial.suggest_int('num_epochs', 5, 30)

    print(f'----------------------TRIAL {trial.number}------------------------')
    print(f'lr: {lr}\nmomentum: {momentum}\nnum_epochs: {num_epochs}\n')
    # 加载预训练的ResNet模型
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(train.class_names))
    model = model.to(train.device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # 训练模型
    model = train.train_model(
        model, criterion, optimizer,
        train.dataloaders, train.dataset_sizes,
        train.device, num_epochs
    )

    torch.save(model.state_dict(), f'lung_xray_densenet121_trial{trial.number}.pth')

    # 验证模型并计算macro F1 score
    f1_macro, f1_weighted, accuracy, average_test_loss, class_f1_scores = test.test_loop(test.dataloaders, model, criterion)

    # 打印超参数和F1 score
    print(f"f1_macro: {f1_macro}, f1_weighted: {f1_weighted}, accuracy: {accuracy}, average_test_loss: {average_test_loss}")
    for class_name, class_f1 in class_f1_scores.items():
        print(f"Class: {class_name}, F1 score: {class_f1}")

    return f1_macro


if __name__ == '__main__':
    # 运行 Optuna 优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # 输出最佳超参数
    print(f'Best trial: {study.best_trial.value}')
    print(f'Best params: {study.best_trial.params}')

