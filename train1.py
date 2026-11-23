from ultralytics import YOLO

if __name__ == '__main__':
    # 先加载你的自定义结构
    model = YOLO(r'D:\cun\yolov13-main\ultralytics\cfg\models\v13\yolov13n-p2-scfem.yaml')
    # 再加载预训练权重，已训练：n-ema、n-scfem、n-ema-scfem、n-p2-scfem
    model.load(r'D:\cun\yolov13-main\yolov13n.pt')
     # 开始训练，这是最终的训练脚本，已训练p2-ema-scfem-v4、n-p2-ema2、n-p2、n、yolov5n、yolov8n、yolov8s、yolov10n、yolov11n、
    model.train(
        data=r'D:\cun\yolov13-main\data.yaml',
        imgsz=960,            # 图像尺寸
        epochs=450,           # 训练轮数
        batch=4,              # 批处理大小
        workers=4,            # 数据加载线程数
        device=0,             # GPU编号，0代表第一块卡你
        optimizer='AdamW',    # 优化器，适合小数据集微调
        lr0=0.003,            # 初始学习率
        lrf=0.01,             # 最终学习率比例
        cos_lr=True,          # 余弦退火学习率
        close_mosaic=80,      # 最后80轮关闭mosaic
        mosaic=0.3,           # 降低mosaic强度
        mixup=0.0,            # 关闭mixup
        copy_paste=0.0,       # 关闭copy_paste
        box=10.0,             # 增强定位损失权重，5轮由10改8
        cls=0.3,              # 单类任务，分类权重可低一些，5轮由0.3改0.6
        dfl=2.0,              # DFL损失权重，5轮由2改1.8
        patience=100,         # 早停
        single_cls=True,      # 单类别 crack
        cache=False,          # 不缓存
        project='runs/train', # 保存目录
        name='n-p2-scfem',      # 本次训练名称，已训练p2-ema-1、p2-ema-cdsa-v2、p2-ema-cdsa-v3、p2-ema-scfem-v4
        resume=False          # 不断点续训
    )