from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    model = YOLO(r"D:\cun\yolov13-main\runs\train\n-ema\weights\best.pt")

    metrics = model.val(
        data=r'D:\cun\yolov13-main\data.yaml',
        imgsz=960,
        batch=4,
        device=0,
        split="val",
        plots=True,
        save_json=True,#已验证n-ema_best_val、n-scfem_best_val、n-ema-scfem_best_val、n-p2-scfem_best_val
        project=r"D:\cun\yolov13-main\runs\val",
        name="n-ema_best_val",#已验证p2-ema-scfem-v4_best_val、n-p2-ema2_best_val、n-p2_best_val
        exist_ok=True,#已验证yolov5n_best_val、yolov8n_best_val、yolov8s_best_val、yolov10n_best_val、yolov11n_best_val
        verbose=True
    )

    print(metrics)


if __name__ == "__main__":
    freeze_support()
    main()