# main.py

import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils, engine
from train import VehicleDetectionDataset, load_category_map
# from transforms import Compose, PILToTensor, ToDtype, RandomHorizontalFlip

def get_transform(train: bool):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# def get_transform(train):
#     transforms = [PILToTensor(), ToDtype(torch.float32, scale=True)]
#     if train:
#         transforms.insert(0, RandomHorizontalFlip(0.5))
#     return Compose(transforms)

def main():
    root       = 'stage-3/week_2/data/r-cnn-data/vehicles_images'
    train_csv  = os.path.join(root, 'train_labels.csv')
    test_csv   = os.path.join(root, 'test_labels.csv')
    cat_txt    = os.path.join(root, 'category.txt')

    # 載入類別 map
    category_map = load_category_map(cat_txt)
    num_classes  = len(category_map) + 1

    # 建立 Dataset & DataLoader
    dataset_train = VehicleDetectionDataset(root + '/train', train_csv, category_map, get_transform(True))
    dataset_test  = VehicleDetectionDataset(root + '/test',  test_csv,  category_map, get_transform(False))
    data_loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True,  collate_fn=utils.collate_fn)
    data_loader_test  = DataLoader(dataset_test,  batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    # 模型初始化
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 優化器與 Scheduler
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 訓練與評估
    num_epochs = 10
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        engine.evaluate(model, data_loader_test,  device)

if __name__ == '__main__':
    main()
