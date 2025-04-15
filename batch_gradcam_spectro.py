# batch_gradcam_spectro.py
# 针对 spectrogram 图像（cqt, cwt）批量生成 GradCAM 可视化图像

import os
import glob
import torch
from tqdm import tqdm
from spectro_ablation_experiments import CausalSpectroNet
from spectro_gradcam_visualizer import generate_gradcam_overlay

def run_gradcam_for_spectrograms():
    root_dir = "Voice"
    output_dir = "gradcam_outputs"
    model_path = "causal_spectro_diagnosis2.pth"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalSpectroNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # ✅ 搜索所有 *Output 文件夹中的 cqt/cwt 图像
    pattern = os.path.join(root_dir, "**", "*Output", "*.png")
    image_paths = sorted([
        p for p in glob.glob(pattern, recursive=True)
        if any(x in p.lower() for x in ["cqt", "cwt"]) and p.endswith(".png")
    ])

    print(f"Total spectrogram images found: {len(image_paths)}")

    for img_path in tqdm(image_paths, desc="Generating GradCAM"):
        json_path = img_path.replace(".png", "_spec.json")
        if not os.path.exists(json_path):
            print(f"[WARN] Missing JSON metadata: {json_path}")
            continue

        filename = os.path.basename(img_path).replace(".png", "")
        out_path = os.path.join(output_dir, f"{filename}_gradcam.png")

        try:
            generate_gradcam_overlay(
                model=model,
                image_path=img_path,
                json_path=json_path,
                target_class_idx=1,
                target_layer_name="features.7",
                output_path=out_path,
                return_cam=False
            )
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")

if __name__ == "__main__":
    run_gradcam_for_spectrograms()
