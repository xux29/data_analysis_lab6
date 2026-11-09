# Lab06 实验指南：Detection & Tracking 论文复现

> 课程：CSE5002《智能数据分析》  
> 任务来源：`lec-6-2025.pdf`  
> 截止日期：见实验平台

---

## 1. 任务描述（来自 lec-6-2025.pdf 的简述）

本实验需要：

1. **精读两篇论文 + 一篇硕士论文**：
   - 在线检测与跟踪（Joint detection & tracking）：`Joint_Multi-Object_Detection_and_Tracking_with_Camera-LiDAR_Fusion_for_Autonomous_Driving.pdf`
   - 离线跟踪（Offline tracking）：`BiTrack_Bidirectional_Offline_3D_Multi-Object_Tracking_Using_Camera-LiDAR_Data.pdf`
   - 硕士论文（综合两篇工作）：`12032943-黄珂邈-面向自动驾驶三维目标感知的在线与离线算法研究.pdf`

2. **运行两篇论文的开源代码**，尽量在 KITTI 数据集（或其子集）上得到结果。

3. **撰写报告（PDF 格式）**：包含复现过程、结果（含数字）、以及评论（comment）。

---

## 2. 两篇论文 & 开源代码定位

两篇论文和硕士论文都是 **Kemiao Huang（黄珂邈）** 的工作，作者本人是南方科技大学 2023 届硕士生，导师是 Hao Qi 副教授。论文均围绕 **3D 多目标跟踪** 任务：JMODT 是 IROS 2021 的在线算法，BiTrack 是 ICRA 2025 的离线算法。两份开源代码都托管在作者 GitHub 上（`Kemo-Huang`），本次实验已经 `git clone --depth 1` 到本目录：

| 论文 | 发表 | 角色 | 代码仓库 | 本地路径 |
|---|---|---|---|---|
| JMODT | IROS 2021 | 在线检测与跟踪 | https://github.com/Kemo-Huang/JMODT | `./JMODT/` |
| BiTrack | ICRA 2025 | 离线双向跟踪 | https://github.com/Kemo-Huang/BiTrack | `./BiTrack/` |
| 硕士论文 | 2023（答辩） | 综述 + 综合实验 | （无独立代码仓库，两篇论文即对应章节） | — |

硕士论文的目录结构对应关系已经梳理清楚：
- **第 3 章**：JMODT（联合三维多目标检测与跟踪）
- **第 4 章**：三维多目标跟踪的离线优化（即 BiTrack 的核心方法）

---

## 3. 仓库结构与代码入口

### 3.1 BiTrack（离线）

入口脚本在仓库根目录，全部使用 `python kitti_*.py $config_path ...` 形式跑：

| 脚本 | 阶段 |
|---|---|
| `kitti_3d_detection.py` | KITTI 3D 目标检测（VoxelRCNN） |
| `kitti_2d_mots.py` | KITTI 2D 实例分割（SpatialEmbedding） |
| `data_processing/crop_points.py` | 裁剪 3D 框内点云 |
| `data_processing/save_img_shapes.py` | 保存图像尺寸 |
| `kitti_2d_3d_det_fusion.py` | **2D-3D 检测融合**（核心创新点之一） |
| `kitti_3d_tracking.py --backward` / `--no-backward` | 正向 / 反向跟踪 |
| `kitti_trajectory_refinement.py` | **轨迹融合 + 单轨迹精修**（核心创新点之二） |
| `TrackEval/scripts/run_kitti.py` | KITTI 官方跟踪评测 |

子模块：
- `tracking/association.py`：相似度计算（IoU / CD / NCD）+ 匈牙利匹配
- `tracking/motion_filters.py`：3D 卡尔曼滤波器（状态量 = `(x, y, z, l, w, h, a, vx, vy, vz)`）
- `tracking/tracker.py`：在线 / 离线双向跟踪主循环
- `tracking/trajectory_clustering_split_and_recombination.py`：双向轨迹拆分-合并（BFS 聚类 + 贪心挑选）
- `tracking/trajectory_completion.py`：线性插值补全 + NMS 防抖
- `tracking/trajectory_refinement.py`：加权尺寸平均 + 高斯过程回归
- `detection/voxel_rcnn/`：3D 检测网络（基于 OpenPCDet / Voxel R-CNN）
- `segmentation/spatial_embeddings/`：2D 实例分割

### 3.2 JMODT（在线）

入口脚本都在 `tools/` 下：

| 脚本 | 阶段 |
|---|---|
| `tools/kitti_converter.py` | KITTI 数据格式转换（tracking → detection） |
| `tools/train.py` | 训练（需 pretrained EPNet 检测权重） |
| `tools/eval.py` | 评测 |

子模块：
- `jmodt/detection/`：检测网络（基于 EPNet / LI-Fusion）
- `jmodt/tracking/kalman.py`：3D 卡尔曼滤波
- `jmodt/tracking/data_association.py`：**MIP（混合整数规划）数据关联**（核心创新点）
- `jmodt/tracking/tracker.py`：在线跟踪主循环
- `jmodt/ops/`：CUDA 算子（`iou3d`、`pointnet2`、`roipool3d`）

---

## 4. 运行环境与依赖

### 4.1 硬件

| 项 | 实际值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| CUDA | 12.1 |
| Python | 3.11（conda env `data_analysis`） |
| PyTorch | 已安装（含 CUDA 支持） |

### 4.2 数据集：**KITTI Tracking**

两份代码都默认在 **KITTI Multi-Object Tracking Benchmark** 上评测，需要：
- 21 个 training sequences（带 GT 标签）
- 29 个 test sequences（无 GT，需提交 KITTI 评测服务器）
- 标定文件 `calib/`、点云 `velodyne/`、图像 `image_02/`、标签 `label_02/`
- 需 KITTI 官网注册才能获取，下载链接：http://www.cvlibs.net/datasets/kitti/eval_tracking.php

> **重要提示**：本机未提前下载 KITTI 数据集，需要：
> 1. 在 KITTI 官网注册账号
> 2. 下载 `data_tracking.zip` 等
> 3. 解压到 BiTrack 期望的 `data/kitti/tracking/` 或 JMODT 期望的 `data/KITTI/tracking/` 目录

> **数据规模预估**：KITTI tracking 的训练集完整下载约 **16 GB**（image_02 占据大部分）；如果不想下载图像只做点云 + 跟踪流程，可以用 tracking-only 模式跳过检测 / 分割部分，但 README 推荐的完整流程需要图像。

### 4.3 模型权重

| 模型 | 来源 | 大小 |
|---|---|---|
| VoxelRCNN 检测权重 | BiTrack 仓库 README 的 Google Drive | ~50 MB |
| SpatialEmbedding 分割权重 | 同上 | ~100 MB |
| JMODT 完整权重（EPNet + ReID） | JMODT README 的 Google Drive | ~30 MB |

> 国内访问 Google Drive 不稳，备用地址见 BiTrack 仓库说明。

### 4.4 编译 CUDA 算子

两个仓库都依赖自定义 CUDA 算子：

```bash
# BiTrack
cd BiTrack && python setup.py develop

# JMODT
cd JMODT && python setup.py develop
```

这两个 `setup.py` 内部都调用 `CUDAExtension` 编译 `.cu` 文件，需要：
- 与 PyTorch 版本兼容的 CUDA Toolkit
- nvcc 编译器

> 当前 `data_analysis` 环境的 PyTorch 编译用的是 CUDA 12.1，应该能匹配 RTX 3050。

### 4.5 Python 依赖

JMODT 的 `requirements.txt`：

```
numpy, easydict, scipy, pyyaml, fire, numba, ortools,
filterpy, munkres, tqdm, tensorboardX, opencv-python, pillow, open3d
```

> `ortools`（混合整数规划求解器）、`filterpy`（卡尔曼滤波）、`munkres`（匈牙利算法）、`open3d`（点云可视化）这几个是 JMODT 必需的；`numba` 需要 JIT 编译，第一次跑会慢一些。

BiTrack 没有单独 `requirements.txt`，但实际需要：
```
numpy, easydict, scipy, pyyaml, tqdm, opencv-python, pillow, filterpy, munkres, open3d, motmetrics (or TrackEval)
```

---

## 5. 推荐复现流程

考虑到 KITTI 数据集下载耗时 + 权重下载 + 编译 CUDA 算子 + GPU 训练检测网络总耗时很长（首次全套可能 **3-5 天**），下面是分优先级方案：

### 方案 A（**推荐**）：最小可行复现 — 跟踪 pipeline

跳过训练检测网络这一步，**使用 KITTI 公开提供的 detection 结果**（或仓库里已有的 `det3d_out/`）：

1. 准备 KITTI tracking 数据（只取 velodyne + calib + label_02）
2. 跑 BiTrack 的 `kitti_3d_tracking.py`（正向 + 反向）
3. 跑 `kitti_trajectory_refinement.py`
4. 跑 `TrackEval/scripts/run_kitti.py` 得到 MOTA / HOTA 数值
5. 与论文 Table II / Table III 对比，记录差距和原因

预期耗时：**半天~1 天**（如果数据已就绪）。

### 方案 B：完整 BiTrack 复现

按 BiTrack README 全部走一遍：3D 检测 → 2D 分割 → 2D-3D 融合 → 双向跟踪 → 精修 → 评测。

预期耗时：**2-4 天**（含下载、编译、推理）。

### 方案 C：JMODT 复现

需要训练；按 JMODT README `tools/train.py` 走起。

预期耗时：**3-5 天**（含训练 50 epoch）。

### 方案 D（**最小代价的兜底**）：纯代码阅读 + 单帧 demo

如果数据集和权重都下不来，**至少做到**：

1. 在 CPU 上 import BiTrack 的 `tracker.py` 和 JMODT 的 `data_association.py`
2. 构造假数据（mock 几个 3D 检测框和预测框），跑通 `association.py` 中的匈牙利匹配
3. 跑通 JMODT 的 MIP 求解（用 mock 数据）
4. 报告里写「环境受限，仅完成了算法逻辑验证，未完成端到端评测」

预期耗时：**半天**。

---

## 6. 实验报告（PDF）大纲

按 `lec-6-2025.pdf` 的要求，PDF 报告建议结构：

1. **封面**：姓名、学号、日期
2. **引言**（1 段）：论文背景，为什么 3D 多目标跟踪重要
3. **论文 1（BiTrack）**：
   - 核心方法总结（点级别 2D-3D 融合 + 双向跟踪 + 轨迹精修）
   - 复现尝试（编译、数据准备、跑通到哪一步）
   - 结果（如果有数字：MOTA/HOTA；没有就报告失败原因）
   - 个人评论（优缺点、与 baseline 对比、有何改进空间）
4. **论文 2（JMODT）**：
   - 核心方法总结（联合检测 + 平行 Re-ID + MIP 数据关联）
   - 复现尝试
   - 结果
   - 评论
5. **硕士论文（黄珂邈）综合**：
   - 两篇工作的关系
   - 与单独的 JMODT / BiTrack 相比有何补充
6. **总结**（1 段）：本次实验最大收获、踩过的坑、可继续做的工作

> PDF 长度建议：**8-15 页**，太短显得没做，太长（>25）会显得堆字数。  
> 配图：至少要有 1 张系统架构图（从论文截图）+ 1 张 KITTI 跟踪可视化（如果跑通）。

---

## 7. 关键参数速查（来自论文）

### BiTrack（论文 Table II + Section V-B）

| 参数 | 值 | 含义 |
|---|---|---|
| `θ_det` | 0.85 | 3D 检测置信度阈值 |
| `α` | 0 | 2D-3D 相似度阈值 |
| `β` | 0.5 | 跟踪相似度阈值 |
| `θ_hit` | 6 | 击中次数，确认轨迹 |
| `θ_miss` | {5, 28} | 漏检次数阈值（新生 vs 已确认） |
| `P₀, Q, R` | 10I, 2I, I | 卡尔曼滤波协方差 |
| `θ_inter` | 4 | 插值滑动窗口 |
| `γ` | 0.35 | 插值去重 NCD 阈值 |
| `τ` | 5.5 | 高斯过程平滑参数 |
| KITTI MOTA | 87.81% | 训练集（论文报告） |
| KITTI HOTA | 84.54% | 训练集（论文报告） |

### JMODT（论文）

| 参数 | 值 | 含义 |
|---|---|---|
| KITTI MOTA | 86.10% | 验证集（论文报告） |
| KITTI AP-Easy | 94.01% | |
| 检测模型 | EPNet | 基础网络 |
| 跟踪器 | KF + MIP | |
| 关联度量 | 3D DIoU | |

---

## 8. 已知难点 & 风险

1. **KITTI 数据集下载**：要注册且网络不稳。
3. **CUDA 编译**：与 PyTorch 版本的兼容性，如果 env 里 PyTorch 是 CPU 版（`+cpu` 后缀），无法编译；当前 `data_analysis` 环境已含 CUDA 版 PyTorch，理论上 OK。
4. **显存**：VoxelRCNN 全量推理需要 ~6 GB 显存，RTX 3050 笔记本版只有 4 GB，**可能 OOM**——这种情况下需要降 batch_size 或用 SpatialEmbedding / PointRCNN 替代。
5. **数据预处理**：BiTrack 的 `crop_points.py` 需要遍历所有帧裁剪点云，几 GB 数据下耗时较长。
6. **Time budget**：完整复现一次 BiTrack 在单机 RTX 3050 上大约 **2-3 天**。

---

## 9. 进度跟踪

| 阶段 | 状态 |
|---|---|
| 精读两篇论文 + 硕士论文 | ✅ 已完成（已通过 pypdf 提取全部文本到 `extracted_papers.txt`） |
| 克隆 BiTrack 仓库 | ✅ 已完成 |
| 克隆 JMODT 仓库 | ✅ 已完成 |
| 阅读两份 README + config.py | ✅ |
| 安装 Python 依赖 | ⬜ 待做 |
| 编译 CUDA 算子 | ⬜ 待做（先跑 `python setup.py develop`） |
| 下载 KITTI 数据 | ⬜ 待做 |
| 下载模型权重 | ⬜ 待做 |
| 跑通 BiTrack 端到端 | ⬜ 待做 |
| 跑通 JMODT 端到端 | ⬜ 待做 |
| 撰写 PDF 报告 | ⬜ 待做 |

---

## 10. 当前文件清单

```
Lab06/
├── guide.md                                           ← 本文件
├── lec-6-2025.pdf                                     ← 课堂讲义（密码保护，未解压）
├── Joint_Multi-Object_Detection_and_Tracking_...pdf   ← 论文 1（JMODT，密码保护）
├── BiTrack_Bidirectional_Offline_...pdf              ← 论文 2（BiTrack，密码保护）
├── 12032943-黄珂邈-...pdf                             ← 硕士论文（密码保护）
├── extracted_papers.txt                               ← pypdf 提取的两篇英文论文全文
├── BiTrack/                                           ← 已克隆的 BiTrack 仓库
└── JMODT/                                             ← 已克隆的 JMODT 仓库
```

> PDF 都是密码保护的（"Authorized licensed use limited to..."），但 `pypdf` 能直接读出正文，已把两篇英文论文的全文写到 `extracted_papers.txt`。硕士论文 79 页尚未提取，后续可以直接用 `pypdf` 按页读。