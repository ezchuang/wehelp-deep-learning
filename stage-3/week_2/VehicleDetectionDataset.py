# main.py

import os
import ast
from typing import cast
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision.transforms.functional as F

# 1. 讀取 category_map
#    格式："category: {'Bus':0, ...}"
def load_category_map(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    mapping_str = text.split('category:')[1].strip()
    return ast.literal_eval(mapping_str)

# 2. collate_fn：將 batch 拆成兩個 list
#    detection API 需要 list of images, list of targets
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# 3. Dataset for vehicle detection
def get_label_column(df, category_keys):
    # 找到那欄所有值都屬於 category_map keys
    for col in df.columns:
        if col.lower() in ('filename', 'xmin', 'ymin', 'xmax', 'ymax'):
            continue
        if set(df[col].unique()) <= set(category_keys):
            return col
    raise KeyError(f"無法辨識 label 欄位, CSV 欄: {df.columns.tolist()}")

class VehicleDetectionDataset(Dataset):
    def __init__(self, images_root, csv_file, category_map, transform=None):
        self.images_root = images_root
        self.df = pd.read_csv(csv_file)
        self.category_map = category_map
        # 固定 box 欄位
        self.box_cols = ['xmin', 'ymin', 'xmax', 'ymax']
        # 自動偵測 label 欄位
        self.label_col = get_label_column(self.df, category_map.keys())
        # transform: PIL -> Tensor
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # load image
        img_path = os.path.join(self.images_root, row['filename'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # boxes
        x1, y1, x2, y2 = [row[c] for c in self.box_cols]
        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        # label
        lbl = row[self.label_col]
        labels = torch.tensor([self.category_map[lbl]], dtype=torch.int64)
        # other target fields
        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        return img, target

# 4. build model
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    predictor: FastRCNNPredictor = cast(FastRCNNPredictor, model.roi_heads.box_predictor)
    # 從 weight 大小取 in_features
    in_features = predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 5. training for one epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        # losses 是 tensor
        losses = torch.stack(list(loss_dict.values())).sum()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    print(f"Epoch {epoch + 1}, loss: {total_loss/len(data_loader):.4f}")

# 6. evaluation
def evaluate(model, data_loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, _ in data_loader:
            imgs = [img.to(device) for img in images]
            outputs = model(imgs)
            for out in outputs:
                scores.extend(out['scores'].cpu().tolist())
    return sum(scores)/len(scores) if scores else 0.0

# 7. visualization
def visualize_predictions(model, data_loader, category_map, device, threshold=0.5, n=2):
    model.eval()
    images, _ = next(iter(data_loader))
    with torch.no_grad():
        outputs = model([img.to(device) for img in images[:n]])
    for i in range(n):
        pil = F.to_pil_image(images[i])
        plt.figure(figsize=(8,8)); plt.imshow(pil); ax=plt.gca()
        for box, lbl, scr in zip(outputs[i]['boxes'], outputs[i]['labels'], outputs[i]['scores']):
            if scr < threshold: continue
            x1,y1,x2,y2 = box.cpu().numpy()
            ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1, fill=False, linewidth=2))
            cls = next(k for k,v in category_map.items() if v==lbl.item())
            ax.text(x1, y1, f"{cls}:{scr:.2f}", bbox=dict(facecolor='white',alpha=0.5))
        plt.axis('off'); plt.show()

# 8. main
def main():
    root = os.path.join('stage-3','week_2','data','r-cnn-data','vehicles_images')
    train_dir = os.path.join(root,'train')
    test_dir  = os.path.join(root,'test')
    train_csv = os.path.join(root,'train_labels.csv')
    test_csv  = os.path.join(root,'test_labels.csv')
    cat_txt   = os.path.join(root,'category.txt')

    category_map = load_category_map(cat_txt)
    num_classes = len(category_map) + 1

    torch.backends.cudnn.benchmark = True

    transform = T.ToTensor()
    train_ds = VehicleDetectionDataset(train_dir, train_csv, category_map, transform)
    test_ds  = VehicleDetectionDataset(test_dir,  test_csv,  category_map, transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False,num_workers=4, pin_memory=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = get_model(num_classes).to(device)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(10):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

    avg_score = evaluate(model, test_loader, device)
    print(f"Average score: {avg_score:.4f}")

    visualize_predictions(model, test_loader, category_map, device)

if __name__ == '__main__':
    main()