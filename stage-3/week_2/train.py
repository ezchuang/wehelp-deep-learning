# train.py

import os
import ast
from typing import cast
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision.transforms.functional as F

# 1. 讀取 category_map
def load_category_map(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    mapping_str = text.split('category:')[1].strip()
    return ast.literal_eval(mapping_str)

# 2. collate_fn
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# 3. helper to detect label column
def get_label_column(df, category_keys):
    for col in df.columns:
        if col.lower() in ('filename','xmin','ymin','xmax','ymax'):
            continue
        if set(df[col].unique()) <= set(category_keys):
            return col
    raise KeyError('無法辨識 label 欄位')

# 4. Dataset for vehicle detection
class VehicleDetectionDataset(Dataset):
    def __init__(self, images_root, csv_file, category_map, transform=None):
        self.images_root = images_root
        self.df = pd.read_csv(csv_file)
        self.category_map = category_map
        self.box_cols = ['xmin','ymin','xmax','ymax']
        self.label_col = get_label_column(self.df, category_map.keys())
        # self.transform = transform or T.ToTensor()
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.images_root, row['filename'])).convert('RGB')

        # 準備 target dict
        boxes = torch.tensor([[row['xmin'], row['ymin'], row['xmax'], row['ymax']]], dtype=torch.float32)
        labels = torch.tensor([self.category_map[row[self.label_col]]], dtype=torch.int64)
        # image_id = torch.tensor([idx])
        image_id = idx
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        # ← 這裡同時傳入 img 與 target
        img_tensor, target = self.transform(img, target)

        return img_tensor, target

# 5. build model
def get_model(num_classes):
    # 使用新版 weights 參數
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    # 明確指定型別並使用 in_features 屬性
    predictor: FastRCNNPredictor = cast(FastRCNNPredictor, model.roi_heads.box_predictor)
    in_features = predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 6. training epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        imgs = [img.to(device, non_blocking=True) for img in images]
        tars = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, tars)
        losses = torch.stack(list(loss_dict.values())).sum()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    print(f"Epoch {epoch}, loss: {total_loss/len(data_loader):.4f}")

# 7. evaluation
def evaluate(model, data_loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, _ in data_loader:
            imgs = [img.to(device, non_blocking=True) for img in images]
            outs = model(imgs)
            for out in outs:
                scores.extend(out['scores'].cpu().tolist())
    return sum(scores)/len(scores) if scores else 0.0

# 8. visualize GT vs Pred
def visualize_gt_pred(dataset, model, category_map, device, out_dir='vis', n=5, thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for idx in range(min(n, len(dataset))):
        img_tensor, target = dataset[idx]
        img_batch = img_tensor.to(device).unsqueeze(0)
        with torch.no_grad():
            pred = model(img_batch)[0]
        pil_img = F.to_pil_image(img_tensor)
        fig, ax = plt.subplots(1, figsize=(8,8))
        ax.imshow(pil_img)
        for box in target['boxes']:
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                                   edgecolor='g', linestyle='--', fill=False, linewidth=2))
        for box, lbl, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            if score < thr:
                continue
            x1, y1, x2, y2 = box.cpu().tolist()
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                                   edgecolor='r', fill=False, linewidth=2))
            cls = next(k for k, v in category_map.items() if v == lbl.item())
            ax.text(x1, y1, f"{cls}:{score:.2f}", color='r',
                    bbox=dict(facecolor='white', alpha=0.5))
        ax.axis('off')
        fig.savefig(os.path.join(out_dir, f"vis_{idx}.png"))
        plt.close(fig)

# 9. main
def main():
    root = os.path.join('stage-3', 'week_2')
    data_root = os.path.join(root, 'data', 'r-cnn-data', 'vehicles_images')
    train_dir = os.path.join(data_root, 'train')
    test_dir  = os.path.join(data_root, 'test')
    train_csv = os.path.join(data_root, 'train_labels.csv')
    test_csv  = os.path.join(data_root, 'test_labels.csv')
    cat_txt   = os.path.join(data_root, 'category.txt')

    category_map = load_category_map(cat_txt)
    num_classes = len(category_map) + 1

    transform = T.ToTensor()
    train_ds = VehicleDetectionDataset(train_dir, train_csv, category_map, transform)
    test_ds  = VehicleDetectionDataset(test_dir, test_csv,  category_map, transform)

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=10,
        pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds,  batch_size=4, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    model = get_model(num_classes).to(device)
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    for epoch in range(1, 11):
        train_one_epoch(model, optimizer, train_loader, device, epoch)

    torch.save(model.state_dict(), f'{root}/model.pth')
    print('Model saved to model.pth')

    loaded_model = get_model(num_classes)
    loaded_model.load_state_dict(torch.load(f'{root}/model.pth'))
    loaded_model.to(device)
    visualize_gt_pred(train_ds, loaded_model, category_map, device, out_dir=f'{root}/vis_train', n=5)

if __name__ == '__main__':
    main()
