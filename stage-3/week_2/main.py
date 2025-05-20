import os
import ast
import matplotlib.pyplot as plt
from typing import cast
from collections import defaultdict
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from pytorch_vision import engine, utils

# load category_map
def load_category_map(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        mapping_str = f.read().split('category:')[1].strip()
    return ast.literal_eval(mapping_str)

# customized dataset
class VehicleDetectionDataset(Dataset):
    def __init__(self, root, csv_file, category_map, transforms):
        self.root = root
        self.category_map = category_map
        df = pd.read_csv(csv_file)

        # 1. find unique filename
        self.img_files = df['filename'].unique().tolist()

        # 2. add all the frames and labels into the List, save to annotations dict
        annots = defaultdict(list)
        labels = defaultdict(list)
        for _, row in df.iterrows():
            annots[row['filename']].append([
                row['xmin'], row['ymin'], row['xmax'], row['ymax'],
                # category_map[row['class']] + 1
                # category_map[row['class']]
            ])
            labels[row['filename']].append([category_map[row['class']] + 1])
        self.annotations = annots
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filename = self.img_files[idx]
        img = Image.open(os.path.join(self.root, filename)).convert('RGB')

        # convert annotations to tenser, and separate them into boxes and labels
        ann = torch.tensor(self.annotations[filename], dtype=torch.float32)
        lab = torch.tensor(self.labels[filename], dtype=torch.int64)
        boxes  = ann[:, :4]
        labels = lab[:, 0]

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': idx,
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        img = tv_tensors.Image(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

# 3. model builder
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # cast predictor
    predictor: FastRCNNPredictor = cast(FastRCNNPredictor, model.roi_heads.box_predictor)
    # get in_features from weight size
    in_features = predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 4. transform builder
def get_transform(train: bool):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.RandomVerticalFlip(0.2))
        # transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        # transforms.append(T.RandomRotation(degrees=10))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def main():
    base = os.path.join('stage-3', 'week_2', "data")
    data_root = os.path.join(base, "r-cnn-data", "vehicles_images")
    cat_txt   = os.path.join(data_root, 'category.txt')
    train_dir = os.path.join(data_root, 'train')
    test_dir  = os.path.join(data_root, 'test')
    train_csv = os.path.join(data_root, 'train_labels.csv')
    test_csv  = os.path.join(data_root, 'test_labels.csv')
    out_dir   = os.path.join(base, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    # data & transforms
    category_map = load_category_map(cat_txt)
    # return 
    ncls = len(category_map) + 1

    ds_train = VehicleDetectionDataset(train_dir, train_csv, category_map, get_transform(True))
    ds_test  = VehicleDetectionDataset(test_dir,  test_csv,  category_map, get_transform(False))
    dl_train = DataLoader(ds_train, batch_size=2, shuffle=True,  collate_fn=utils.collate_fn)
    dl_test  = DataLoader(ds_test,  batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    # model & optimizer
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = get_model(ncls).to(device)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    # optimizer = torch.optim.AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=0.001, weight_decay=0.0001
    # )
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)


    train_losses = []
    mAPs = []

    # train & eval
    EPOCHS = 22
    for epoch in range(1, EPOCHS+1):
        metric_logger = engine.train_one_epoch(model, optimizer, dl_train, device, epoch, print_freq=10)
        train_loss = metric_logger.loss.global_avg
        train_losses.append(train_loss)

        print(f"Epoch: [{epoch}] learning rate:", optimizer.param_groups[0]['lr'])
        scheduler.step()

        coco_evaluator = engine.evaluate(model, dl_test, device)
        mAP = coco_evaluator.coco_eval['bbox'].stats[0]
        mAPs.append(mAP)

    # 迴圈結束後，畫出並存檔
    epochs = list(range(1, EPOCHS+1))

    # 1. Train Loss 曲線
    plt.figure()
    plt.plot(epochs, train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'train_loss_curve.png'))
    plt.close()

    # 2. mAP 曲線
    plt.figure()
    plt.plot(epochs, mAPs)
    plt.xlabel('Epoch')
    plt.ylabel('mAP@[0.50:0.95]')
    plt.title('Validation mAP over Epochs')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'val_mAP_curve.png'))
    plt.close()

    # save model
    mp = os.path.join(out_dir, 'model.pth')
    torch.save(model.state_dict(), mp)
    print(f"Model saved to {mp}")

if __name__ == '__main__':
    main()
