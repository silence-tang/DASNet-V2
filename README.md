# DASNet-V2
An improved version of DASNet, achieving 0.96+ in F1_score.


## 一、前言
- 参考论文：DASNet: Dual attentive fully convolutional siamese networks for change detection in high-resolution satellite images
- 参考代码：https://github.com/lehaifeng/DASNet
- 说明：本人复现该论文时发现开源代码有部分写错的地方，均已修正

## 二、改进思路
-  整个模型给人头重脚轻的感觉：一开始用很重的模块只是为了算一个融合空间和通道注意力的embedding map，可以理解是为了获得更为有效的representation，但是后续模块就比较弱，直接计算了ground truth和embedding map之间的loss，这可能太粗糙了。而且**源码中是用32x32的尺度算loss，这个过程中ground truth的精度其实已经被丢失了，因此模型学出来的知识可能只够用于检测低分辨率图像**。
-  改进思路：借鉴语义分割中的encoder-decoder结构，一方面可以把embedding map逐步由32x32上采样到256x256，然后计算一个loss，另一方面可以引入多尺度loss，在decoder不同layer的输出feature map上与ground truth计算loss，最后把多层loss联合起来优化网络，使得模型可以学到多尺度的信息。此外，由于变化检测任务其实本质上就是像素级别的分类的任务，本质上和语义分割没有区别（语义分割是单点像素的多分类任务，而变化检测是特殊的语义分割，即单点像素处的二分类任务），因此语义分割那块的所有模型都可以拿来借鉴。
-  注：当前版本暂未联合Decoder的multi-layer losses进行训练，仅利用了Decoder最后一层输出的256x256进行训练。

## 三、其他优化
-  √ 每1000个batch都输出P-R图+metric.json，命名方式为`epoch_batch_idx_P_R.jpg`和`epoch_batch_idx_metric.json`。
-  √ 每20个batch，将当前的epoch, batch_id, f_score, best f_score保存到log中。
-  √ 每20个batch打印learning_rate至控制台。
-  √ 最佳模型保存的命名需要带上`epoch_batchid`，便于查找。
-  √ 命令行参数加入--start_epoch --resume以实现训练的暂停与继续。
-  √ 优化了checkpoints/目录结构。

## 四、配置
- 硬件：单卡 NVIDIA RTX 2080Ti
- Requirements: 参考原论文的Github项目
- 数据集：Change Detection Dataset
- 预训练backbone：resnet50-19c8e357.pth
- Train from scratch：进入DASNet-V2目录，直接在命令行运行`python train.py`（可根据需要自行添加其他命令行参数）


## 五、注意点

- 最佳模型路径在cfg文件夹里，linux/windows需要设置相应的路径。
- labels必须为0/1二值单通道图，处理完之后要生成新的train1.txt, val1.txt, test1.txt，其中mask的路径为OUT1，然后改cfg。
- max f_score和AUC基本成正比，因此可以用它衡量模型的性能优劣，也即是可以用max f_score来筛出best model。
- 原始的输入图像经过了减去均值的中心化操作。

## 六、实验结果
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

## 七、其他实验
均在DASNet-Decoder基础上做实验，为了减小单次实验的时长，可以减小epoch数为45（后面再提升也只是0.2左右）

- **空白对照**
  - **SiamseNet+Decoder，Distance Metric = $L_2$**
  - 训练 80 epoch，0-20 epoch lr = 1e-4，21-40 epoch lr = 5e-5，41-60 epoch lr = 1e-5，61-70 epoch lr = 5e-6，71-80 epoch lr = 1e-6
  - **best_max_f = 0.9566**
  - best_epoch:  76，best_batch_idx:  2499
- **EXP1——修改损失函数中距离的度量标准为余弦相似度，同时修改双边阈值为$m_1=0.1, m_2=0.8$，看看效果升还是降。同时可以观察用余弦相似度计算的热力图**
  - **SiamseNet+Decoder，Distance Metric = Cosine Sim**
  - 训练 70 epoch，0-20 epoch lr = 1e-4，21-40 epoch lr = 5e-5，41-60 epoch lr = 1e-5，61-70 epoch lr = 5e-6
  - **best_max_f = 0.9601**
  - best_epoch:  63，best_batch_idx: 2000

![8c5MV.png](https://i.328888.xyz/2023/01/31/8c5MV.png)
![8cscb.png](https://i.328888.xyz/2023/01/31/8cscb.png)

上图中，前两行是cossim的结果，第3行是使用原始$L_2$loss的结果。不难看出，在使用cossim作为损失函数的计算范式后，变化检测结果的可视化结果也是相比原来的更令人满意的。具体体现为：少了很多斑块状噪声块，而且变化区域和不变区域的颜色对比也更加鲜明，这表示改进过的模型能够更加“坚决”地划分出变与不变的区域。

- **EXP2——损失函数有三项，$Loss = λ_1L_{sa} + λ_2L_{ca} + λ_3L_{saca}$，在实验中发现这三项loss总是非常接近的，应该只用最后的$L_{saca}$也行？**
  - **SiamseNet+Decoder，Distance Metric = $L_2$**，$Loss = L_{saca}$
  - 训练 80 epoch，0-20 epoch lr = 1e-4，21-40 epoch lr = 5e-5，41-60 epoch lr = 1e-5，61-70 epoch lr = 5e-6，71-80 epoch lr = 1e-6
  - **best_max_f = 0.9546**
  - best_epoch:  68，best_batch_idx: 2000

![8miyy.png](https://i.328888.xyz/2023/01/31/8miyy.png)

通过结果，我们可以看出，即使将总损失替换为$L_{sasc}$，也不会对模型性能造成太大的损害，F1_score仅仅下降了0.2。

- **EXP3——消融实验**
  - 无SAM、无CAM，其他不变
   - **SiamseNet+Decoder，Distance Metric = $L_2$**
   - 训练 80 epoch，0-20 epoch lr = 1e-4，21-40 epoch lr = 5e-5，41-60 epoch lr = 1e-5，61-70 epoch lr = 5e-6，71-80 epoch lr = 1e-6
   - **best_max_f = 0.9543**
   - best_epoch:  75，best_batch_idx:  2499

![8mb7o.png](https://i.328888.xyz/2023/01/31/8mb7o.png)

- 无SAM、有CAM，其他不变：可以不做实验，因为发现即使SAM、CAM都去掉，性能也只是下降了0.2左右
- 有SAM、无CAM，其他不变：可以不做实验，因为发现即使SAM、CAM都去掉，性能也只是下降了0.2左右

## 八、实验结论
- 加上优化设计的decoder结构后，原始的DASNet性能得到了显著提升（F1_score从0.92+提升至0.95+）
- 将损失函数的计算范式从$L_2$loss改为余弦相似度，DASNet F1_score可达0.96+
- 通过消融实验发现：原始论文中联合三个feature map的loss计算最终损失其实对结果的提升效果不大，只利用CAM+SAM的融合损失作为总损失也不会下降多少性能（下降0.2左右）
- 通过消融实验发现：原始论文中的亮点（CAM、SAM机制）在具有decoder结构下的DASNet网络中无法对提升性能起到显著作用。若将这两个模块去掉，纯粹使用ResNet backbone抽取出来的特征去过一遍decoder，然后进行后续的loss计算，F1_score也只是下降了0.2左右。
- 当去掉SAM、CAM模块后，平均显存使用量约为7.5G，而加上这两个模块后，平均显存使用量接近9G。这说明DASNet-V2的encoder-decoder+无CAM/SAM结构，能够在减小计算量的情况下超越原论文中汇报的F1_score（0.9543 > 0.9299）。


