# DASNet-V2
An improved version of DASNet, achieving 0.96+ in F1_score.

## 前言
- 参考论文：DASNet: Dual attentive fully convolutional siamese networks for change detection in high-resolution satellite images
- 参考代码：https://github.com/lehaifeng/DASNet
- 说明：本人复现该论文时发现开源代码有部分写错的地方，均已修正

## 改进思路
-  整个模型给人头重脚轻的感觉：一开始用很重的模块只是为了算一个融合空间和通道注意力的embedding map，可以理解是为了获得更为有效的representation，但是后续模块就比较弱，直接计算了ground truth和embedding map之间的loss，这可能太粗糙了。而且**源码中是用32x32的尺度算loss，这个过程中ground truth的精度其实已经被丢失了，因此模型学出来的知识可能只够用于检测低分辨率图像**。
-  改进思路：借鉴语义分割中的encoder-decoder结构，一方面可以把embedding map逐步由32x32上采样到256x256，然后计算一个loss，另一方面可以引入多尺度loss，在decoder不同layer的输出feature map上与ground truth计算loss，最后把多层loss联合起来优化网络，使得模型可以学到多尺度的信息。此外，由于变化检测任务其实本质上就是像素级别的分类的任务，本质上和语义分割没有区别（语义分割是单点像素的多分类任务，而变化检测是特殊的语义分割，即单点像素处的二分类任务），因此语义分割那块的所有模型都可以拿来借鉴。
-  注：当前版本暂未联合Decoder的multi-layer losses进行训练，仅利用了Decoder最后一层输出的256x256进行训练。

## 其他优化
-  √ 每1000个batch都输出P-R图+metric.json，命名方式为`epoch_batch_idx_P_R.jpg`和`epoch_batch_idx_metric.json`。
-  √ 每20个batch，将当前的epoch, batch_id, f_score, best f_score保存到log中。
-  √ 每20个batch打印learning_rate至控制台。
-  √ 最佳模型保存的命名需要带上`epoch_batchid`，便于查找。
-  √ 命令行参数加入--start_epoch --resume以实现训练的暂停与继续。
-  √ 优化了checkpoints/目录结构。

## 实验
- 硬件：单卡 NVIDIA RTX 2080Ti
- Requirements: 参考原论文的Github项目
- 数据集：Change Detection Dataset
- 预训练backbone：resnet50-19c8e357.pth
- Train from scratch：进入DASNet-V2目录，直接在命令行运行`python train.py`（可根据需要自行添加其他命令行参数）

### DASNet
- 基础模型
- 训练 60 epoch，前 40 epoch lr=1e-4，后 20 epoch lr = 1e-5
- **best_max_f = 0.9299360653186844**
- best_epoch:  49
- best_batch_idx:  500
![VnwCy.png](https://i.328888.xyz/img/2022/12/04/VnwCy.png)

### DASNet-V2
- SiamseNet+Decoder
- Decoder = 3 * (TransposedConv + BN+ ReLu)
- 训练 80 epoch，0-20 epoch lr = 1e-4，21-40 epoch lr = 5e-5，41-60 epoch lr = 1e-5，61-70 epoch lr = 5e-6，71-80 epoch lr = 1e-6
- batch size = 4
- **best_max_f = 0.9566290556389714**
- best_epoch:  76
- best_batch_idx:  2499
![Vn2X5.png](https://i.328888.xyz/img/2022/12/04/Vn2X5.png)

### DASNet-V2 (replace l_2 loss with cossim loss)
- cossim loss，双边阈值为m_1=0.1, m_2=0.8
- SiamseNet+Decoder
- 训练 70 epoch，0-20 epoch lr = 1e-4，21-40 epoch lr = 5e-5，41-60 epoch lr = 1e-5，61-70 epoch lr = 5e-6
- **best_max_f = 0.9601**
- best_epoch:  63，best_batch_idx: 2000
![92@GBU~JQ_`9TE_T_HE__UF.png](https://img1.imgtp.com/2022/12/04/iz3oiXoG.png)

### 实验结论
- 当使用encoder-decoder结构时，原论文中提到的SAM、CAM机制对于F1_score的提升十分微小
- 原论文使用l_sa, l_ca, l_saca之和作为最终损失，但在实验过程中发现这三个loss到后面越来越接近，测试发现只用l_saca即可
