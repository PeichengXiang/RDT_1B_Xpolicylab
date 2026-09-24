# RDT_1B_Xpolicylab

XPolicyLab 里的 RDT-1B 适配，包含 EgoVLA（H1 + Inspire，38 维）和 SparkArena / Tianji 的训练与评估入口。

EgoVLA 是单视角 benchmark。发布数据里只有主视角 `cam_head`；旧转换把这张图硬链接进了左右腕部。训练读取和评估默认都不再使用这份副本，而是把两个腕部槽填成与主视角同尺寸的黑图（`RDT_CAMERA_MODE=black_wrist`）。旧的复制主视角检查点仍可通过 `RDT_CAMERA_MODE=main_replicated` 显式打开。
