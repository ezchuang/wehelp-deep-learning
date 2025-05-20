import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision.transforms as T
import torchvision.transforms.functional as F

# 從 main.py 匯入必要函式與類別
from train import (
    load_category_map,
    VehicleDetectionDataset,
    get_model,
    collate_fn,
    evaluate as evaluate_model,
)


def visualize_predictions(dataset, model, category_map, device, out_dir='vis_test', n=5, thr=0.5):
    """
    在測試集上執行預測並將結果與 Ground Truth 一併視覺化
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for idx in range(min(n, len(dataset))):
        img_tensor, target = dataset[idx]
        img_batch = img_tensor.to(device).unsqueeze(0)
        with torch.no_grad():
            preds = model(img_batch)[0]

        # 將 tensor 轉回 PIL Image
        pil_img = F.to_pil_image(img_tensor)
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(pil_img)

        # 畫出 Ground Truth
        for box in target['boxes']:
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   edgecolor='g', linestyle='--', fill=False, linewidth=2))

        # 畫出預測框與標籤
        for box, lbl, score in zip(preds['boxes'], preds['labels'], preds['scores']):
            if score < thr:
                continue
            x1, y1, x2, y2 = box.cpu().tolist()
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   edgecolor='r', fill=False, linewidth=2))
            cls = next(k for k, v in category_map.items() if v == lbl.item())
            ax.text(x1, y1, f"{cls}:{score:.2f}", color='r',
                    bbox=dict(facecolor='white', alpha=0.5))

        ax.axis('off')
        fig.savefig(os.path.join(out_dir, f"vis_test_{idx}.png"))
        plt.close(fig)


def main():
    # 設定路徑
    root = os.path.join('stage-3', 'week_2')
    data_root = os.path.join(root, 'data', 'r-cnn-data', 'vehicles_images')
    test_dir = os.path.join(data_root, 'test')
    test_csv = os.path.join(data_root, 'test_labels.csv')
    cat_txt = os.path.join(data_root, 'category.txt')

    # 載入 category map 與模型結構
    category_map = load_category_map(cat_txt)
    num_classes = len(category_map) + 1
    model = get_model(num_classes)

    # 載入訓練好的權重
    state_dict_path = os.path.join(root, 'model.pth')
    model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 建立測試集與 DataLoader
    transform = T.ToTensor()
    test_dataset = VehicleDetectionDataset(test_dir, test_csv, category_map, transform)
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=collate_fn
    )

    # 執行評估
    mean_score = evaluate_model(model, test_loader, device)
    print(f"Test set mean confidence score: {mean_score:.4f}")

    # 視覺化前 n 張影像結果
    visualize_predictions(test_dataset, model, category_map, device, out_dir=os.path.join(root, 'vis_test'), n=5, thr=0.5)


if __name__ == '__main__':
    main()
