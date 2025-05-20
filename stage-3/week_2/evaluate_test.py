import os
import torch
from torch.utils.data import DataLoader
from pytorch_vision import engine, utils
from main import VehicleDetectionDataset, load_category_map, get_transform, get_model
from PIL import ImageDraw, ImageFont
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def run_evaluation():
    base = os.path.join('stage-3', 'week_2', 'data')
    data_root = os.path.join(base, "r-cnn-data", "vehicles_images")
    test_dir  = os.path.join(data_root, 'test')
    test_csv  = os.path.join(data_root, 'test_labels.csv')
    cat_txt   = os.path.join(data_root, 'category.txt')
    model_path = os.path.join(base, 'outputs', 'model.pth')

    category_map = load_category_map(cat_txt)
    num_classes  = len(category_map) + 1
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_ds = VehicleDetectionDataset(test_dir, test_csv, category_map, get_transform(False))
    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False,
        pin_memory=True, collate_fn=utils.collate_fn
    )

    print("Running evaluation on test set...")
    coco_evaluator = engine.evaluate(model, test_loader, device)
    print("Evaluation completed.")

    vis_dir = os.path.join('stage-3', 'week_2', 'data', 'outputs', 'vis_test')
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    count = 0
    total = 0
    for idx in range(len(test_ds)):
        img_t, target = test_ds[idx]
        with torch.no_grad():
            pred = model([img_t.to(device)])[0]
        pil_img = F.to_pil_image(img_t)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()

        # Ground Truth
        for box in target['boxes']:
            x1, y1, x2, y2 = map(int, box.tolist())
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)

        # Pred Result
        for box, lbl, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box.cpu().tolist())
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            cls = next(k for k, v in category_map.items() if v + 1 == lbl.item())
            draw.text((x1, y1), f"{cls}:{score:.2f}", fill='red', font=font)
        pil_img.save(os.path.join(vis_dir, f"test_{idx}.jpg"))
    print(f"Visualizations saved to {vis_dir}")

if __name__ == '__main__':
    run_evaluation()
