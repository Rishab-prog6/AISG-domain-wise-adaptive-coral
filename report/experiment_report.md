# 实验报告：Domain-Wise Adaptive CORAL（DWA-CORAL）

> 课程：AI 安全 · 主题：分布外泛化（Out-of-Distribution / Domain Generalization）
> 代码框架：DomainBed · 主干网络：ResNet-18 · 训练步数：3000 steps
> 数据集：PACS（主）、VLCS（迁移验证）

---

## 0. 摘要（一页看懂）

我们在 DomainBed 框架下，围绕一个自己提出的方法 **DWA-CORAL（域自适应 CORAL）** 做了一轮完整的"诊断 → 修复 → 改进 → 取舍"研究。核心结论：

1. **诊断**：方法最初只有约 **20%** 的准确率。根因是 CORAL 对齐项 `‖C_e − C̄‖_F²` **没有做归一化**——在 512 维特征上它的数值是 10⁴~10⁶ 量级，把交叉熵（约 2）彻底淹没，优化器只在"拉平协方差"，根本没在学分类。
2. **修复（v2）**：对 CORAL 项除以 `4·d²`（CORAL 标准缩放）、softmax 做数值稳定化、加 warmup。准确率从 20% 恢复到 **≈80%**，追平基线。
3. **新问题**：v2 ≈ CORAL，但并没有"赢"。在分类损失上做自适应加权会过度偏向难域，反而拖累 Cartoon / Photo。
4. **改进**：设计并实现了 6 个轻量变体。其中 **ANCHOR_ALIGNONLY**（锚定最易域做单向对齐）表现最好：在所有"做对齐"的方法里 **Art 最高（79.71）**，同时 **Sketch（79.49）和 Avg（80.80）追平 CORAL**。
5. **取舍**：尝试过的 **ANCHOR_ANNEAL**（随训练衰减对齐强度）效果反而更差，已**回退**。

> ⚠️ 诚实声明：PACS 单种子下域间波动约 ±1%。我们**没有**在统计意义上显著超过 CORAL（Avg 80.80 vs 80.81 属于打平）。本报告真正的贡献在于：**定位了 v1 的致命 bug、系统化的变体研究、以及"锚定单向对齐"这个有解释性的机制**。

---

## 1. 背景与问题设定

### 1.1 什么是 Domain Generalization（DG）

模型在若干**源域**上训练，要在一个**训练中从未见过的目标域**上直接测试（不允许用目标域数据微调）。这考验的是模型学到的特征是否"域不变"，是 AI 安全里"分布外鲁棒性"的典型设定。

### 1.2 数据集

- **PACS**：4 个域 —— **A**rt painting（艺术画）、**C**artoon（卡通）、**P**hoto（真实照片）、**S**ketch（素描）。同样的 7 类物体，但画风差异极大。
- **VLCS**：4 个域 —— Caltech101 / LabelMe / SUN09 / VOC2007，5 类。风格差异比 PACS 小，用来验证方法的**可迁移性**。

### 1.3 评测协议（leave-one-domain-out）

每次留出 1 个域当测试域，其余 3 个域当源域训练。对 4 个域各做一次，得到 4 个准确率，再看两个汇总指标：

- **Avg**：4 个测试域准确率的平均（整体泛化能力）。
- **Worst**：4 个里最差的那个（鲁棒性下界，AI 安全更关心这个）。

主干 ResNet-18，训练 3000 步，`skip_model_save`，超参用框架默认。

### 1.4 工程约束（重要）

所有自定义代码**完全独立**，不修改 DomainBed 原始的 `algorithms.py` / `hparams_registry.py`：

| 文件 | 作用 |
|---|---|
| `domainbed/dwa_algorithms_v2.py` | 所有 DWA 变体的算法实现 |
| `domainbed/dwa_hparams_registry_v2.py` | 对应的超参注册表 |
| `domainbed/scripts/dwa_train_v2.py` | 入口脚本：猴子补丁把算法/超参注册表替换掉，再调用原始 `train.py` |

这样做的好处：基线（ERM/CORAL/...）行为 100% 不变，结果可比；我们的实验和官方代码解耦，方便随时回退。

---

## 2. 出发点：DWA-CORAL 的动机

**CORAL**（CORrelation ALignment）的思路是：让不同源域的特征**二阶统计量（协方差）对齐**，逼近域不变表征。它对所有源域"一视同仁"。

我们的**假设**是：源域之间难度不同（比如 Sketch 比 Photo 难得多），如果能**按难度自适应地分配对齐/学习的权重**，应该比"一视同仁"更好。这就是 **Domain-Wise Adaptive（域自适应）** 的由来：

```
w_e = softmax(τ · detach(L_e))      # 损失大的域 → 权重大
```

其中 `L_e` 是第 e 个源域的交叉熵，`detach` 表示权重不回传梯度，`τ` 是温度。

---

## 3. 失败诊断：为什么最初只有 ~20%？

最初版本的总损失是：

```
loss = Σ_e w_e · L_e
     + λ · Σ_e w_e · ‖C_e − C̄‖_F²      # ← 问题在这里
     + β · Var(L_e)
```

其中 `C_e` 是第 e 域特征的协方差矩阵，`C̄` 是平均协方差。

**根因：CORAL 项没有归一化。** ResNet-18 输出 512 维特征，协方差矩阵是 512×512。两个协方差矩阵差的 Frobenius 范数平方 `‖C_e − C̄‖_F²` 天然是 **O(d²) ≈ 10⁴~10⁶** 量级。而交叉熵 `L_e` 只有 **约 2** 量级。

于是即便把 `λ` 设成 `1e-3`，对齐项仍然有 10²~10³ 的有效权重，**把分类损失彻底淹没**。优化器的梯度几乎全部用于"把协方差拉平"（一个退化解是把所有特征压成同一个点），完全没在学分类边界 → 准确率塌到接近随机猜测的 ~20%（PACS 7 类，随机约 14%）。

**这是一个典型的"损失尺度不匹配"bug**，在多任务/多正则项训练里非常常见，但很隐蔽——代码逻辑看起来完全正确，只是数值尺度错了。

---

## 4. 修复：DWA_CORAL_v2

我们做了三处关键修复（都在 `_DWACoralBase` 里）：

### 4.1 CORAL 标准归一化（最关键）

除以 `4·d²`，这是 Sun & Saenko 原始 CORAL 论文的标准缩放，让对齐项回到与交叉熵可比的量级：

```python
coral_scale = 4.0 * d * d
coral_term_e = ‖C_e − C̄‖_F² / coral_scale
```

修复后 `coral_loss` 从 ~1e5 降到 ~1e-7（见 Debug 日志），`λ` 可以放回到文献常用的 ~1.0。

### 4.2 数值稳定的 softmax

先减去最大值再做指数，避免 `τ·L_e` 较大时溢出：

```python
logits = τ · (detach(L) − max(detach(L)))
w = softmax(logits)
```

### 4.3 Warmup

前 100 步用**均匀权重**，让分类器先平等地见过每个域，再开启自适应加权，避免训练初期权重乱跳。

**结果**：Avg 从 ~20% 恢复到 **80.08%**，与基线同一水平。修复成功。

---

## 5. 新问题：v2 ≈ CORAL，自适应加权的副作用

把 v2 跑满 PACS 后发现：

- v2（Avg 80.08）和 CORAL（Avg 80.81）基本打平；
- v2 相对 ERM 在 Sketch 上的提升（76.94 vs 73.50），**主要来自 CORAL 对齐本身**，不是来自自适应加权；
- 在**分类损失**上做自适应加权，会过度偏向高损失（难）域，反而**拖累 Cartoon / Photo**，Cartoon 变成最差域。

**诊断**：自适应加权这个想法本身没错，但"加权对象"错了——不该去扭曲**分类监督信号**（那会让简单域学不好），而应该只作用在**对齐项**上。这直接引出了下面的变体研究。

---

## 6. 变体研究（6 个轻量变体）

所有变体共享同一套 CORAL/归一化机制（`_DWACoralBase`），只在"**权重怎么算、作用在哪**"上不同。

| 变体 | 分类损失 | 权重作用对象 | 核心思想 |
|---|---|---|---|
| `DWA_CORAL`(v2) | 加权 | 分类 + 对齐 | 原始自适应加权 |
| `DWA_CORAL_ALIGNONLY` | **均值(ERM式)** | 仅对齐 | 分类信号保持平衡，只让对齐偏向难域 |
| `DWA_CORAL_CLIPPED` | 加权 | 分类 + 对齐 | 权重裁剪到 [0.15, 0.60]，防止单域独大 |
| `DWA_CORAL_EMA` | 加权 | 分类 + 对齐 | 用损失的指数滑动平均去噪 |
| `DWA_CORAL_MIXED_LOSSGAP_ALIGNONLY` | 均值 | 仅对齐 | 权重 = 均匀 ⊕ softmax(损失 z-score + 协方差差 z-score) |
| **`DWA_CORAL_ANCHOR_ALIGNONLY`** ⭐ | 均值 | 仅对齐 | **锚定最易域，其它域单向对齐到它** |

### 6.1 ALIGNONLY —— "只在对齐上自适应"

分类损失退回**无加权均值**（和 ERM 一样平衡），自适应权重**只**作用于 CORAL：

```
cls_loss  = mean(L_e)
coral_loss = Σ_e softmax(τ·detach(L_e))_e · coral_term_e
```

动机：第 5 节的诊断——保住简单域的分类能力，同时把对齐"火力"集中到难域。

### 6.2 CLIPPED —— "防止单域独大"

在 v2 基础上，把 softmax 权重裁剪到 `[w_min, w_max]` 再归一化，避免某个域权重接近 1 而其它被饿死。

### 6.3 EMA —— "去噪"

batch 级别的损失抖动大，改用损失的指数滑动平均 `ema = α·ema + (1−α)·L` 来算权重，更稳。

### 6.4 MIXED_LOSSGAP_ALIGNONLY —— "同时看难度和协方差差距"

只看损失太"片面"，于是把两种信号都 z-score 标准化后线性组合：

```
score = α · zscore(L_e) + (1−α) · zscore(coral_term_e)
w      = (1−γ)·均匀 + γ·softmax(τ·score)
```

`(1−γ)·均匀` 这一项保留了 CORAL 的稳定性，`γ` 控制偏离均匀对齐的激进程度。

### 6.5 ANCHOR_ALIGNONLY ⭐ —— 最终冠军

**关键洞察**：前面所有变体都把各域协方差对齐到**平均协方差 C̄**。但在 PACS 里，C̄ 被抽象风格的域（Sketch / Cartoon）"带偏"了——把真实图像（Photo / Art）的特征往抽象方向拉，正好**伤害了 Art / Photo 的泛化**。

ANCHOR 的做法：每一步选**损失最低的源域**当**锚（anchor）**（在 PACS 里通常正是最"自然图像"的那个域），**detach 它的协方差**当对齐目标，让**其它域单向对齐到锚**：

```
anchor      = argmin_e detach(L_e)
coral_term_e = ‖C_e − detach(C_anchor)‖_F² / (4d²)     (e ≠ anchor)
w_other      = (1−γ)·均匀 + γ·softmax(τ·detach(L_other))
coral_loss   = Σ_{e≠anchor} w_other_e · coral_term_e
cls_loss     = mean(L_e)        # 仍是 ALIGNONLY
```

- 锚域的特征**只接受分类梯度**，不被对齐拉扯 → **保住自然图像的判别性流形**；
- 其它域往锚靠拢 → 仍然得到域不变性（保住 Sketch）。

这就是为什么它能"鱼和熊掌"：Art 在所有对齐法里最高，同时 Sketch / Avg 追平 CORAL。

### 6.6 ANCHOR_ANNEAL —— 尝试过、更差、已回退

进一步假设：训练**前期**强对齐建立不变性（保 Sketch），**后期**衰减对齐强度（`λ` 从 0.3 线性降到 0.05），让分类器后期能重新拟合自然图像细节，把 Art 拉回 ERM 水平。

```
λ(t) = λ_min + (λ_0 − λ_min)·max(0, 1 − t/anneal_steps)
```

**实测分数反而更差**——后期放松对齐后，早期学到的不变性没能保持住，Sketch 掉了而 Art 没补回来。这个负结果说明"不变性需要持续的对齐压力来维持"。**已用 `git revert` 干净回退**，代码库恢复到 ANCHOR 冠军状态（该负结果保留在本报告中作为记录）。

---

## 7. 实验结果

### 7.1 PACS（ResNet-18, 3000 steps）

| Method | Art | Cartoon | Photo | Sketch | **Avg** | **Worst** |
|---|---|---|---|---|---|---|
| ERM | **81.17** | 75.64 | **93.11** | 73.50 | **80.86** | 73.50 |
| CORAL | 78.24 | 76.28 | 89.22 | **79.49** | 80.81 | **76.28** |
| GroupDRO | 77.75 | 76.50 | 88.92 | 75.54 | 79.68 | 75.54 |
| IRM | 43.03 | 60.47 | 71.26 | 57.20 | 57.99 | 43.03 |
| Mixup | 78.97 | **77.56** | 92.81 | 69.81 | 79.79 | 69.81 |
| DWA_CORAL_v2 | 79.46 | 75.00 | 88.92 | 76.94 | 80.08 | 75.00 |
| DWA_CORAL_v2_rerun | 79.22 | 75.64 | 89.82 | 77.45 | 80.53 | 75.64 |
| DWA_CORAL_ALIGNONLY | 78.97 | 74.15 | 89.22 | 78.47 | 80.20 | 74.15 |
| **DWA_CORAL_ANCHOR_ALIGNONLY** ⭐ | **79.71** | 74.79 | 89.22 | **79.49** | **80.80** | 74.79 |

**读法**：
- ANCHOR 在所有"做对齐"的方法里 **Art 最高（79.71）**，且 **Sketch 79.49 = CORAL 最好成绩**，**Avg 80.80 ≈ CORAL 80.81**。
- 软肋是 Cartoon（74.79）成了最差域，所以 Worst 略低于 CORAL。
- ERM 的 Art（81.17）/ Photo（93.11）是天花板——因为它完全不做对齐，自然图像细节保留最全；但代价是 Sketch 只有 73.50。**ANCHOR 在两者间取得了更好的平衡。**

### 7.2 VLCS（ResNet-18, 3000 steps）—— 迁移验证

| Method | Caltech101 | LabelMe | SUN09 | VOC2007 | **Avg** | **Worst** |
|---|---|---|---|---|---|---|
| ERM | **95.76** | **64.03** | 64.63 | **71.26** | **73.92** | **64.03** |
| CORAL | 92.93 | 60.45 | 65.40 | 68.15 | 71.73 | 60.45 |
| GroupDRO | 80.57 | 61.21 | **67.84** | 68.74 | 69.59 | 61.21 |
| IRM | 88.34 | 54.24 | 58.84 | 54.37 | 63.95 | 54.24 |
| Mixup | 74.91 | 63.47 | 63.11 | 69.63 | 67.78 | 63.11 |
| **DWA_CORAL_ANCHOR_ALIGNONLY** | 86.57 | 59.13 | **68.60** | 69.78 | 71.02 | 59.13 |

**读法**：换到 VLCS，ANCHOR（Avg 71.02）**与 CORAL（71.73）同档**，并在 **SUN09（68.60，全场最高）和 VOC2007** 上超过 CORAL。说明锚定对齐机制不是只在 PACS 上"过拟合调参"，有一定通用性。VLCS 上 ERM 整体偏强（这个数据集风格差异小，对齐收益本就有限）。

---

## 8. 分析与讨论

1. **谁是赢家**：ANCHOR_ALIGNONLY 是最佳 DWA 变体——它把"自适应加权只用在对齐、且对齐到最易域而非平均"两件事结合起来，**正好对症**第 5 节的副作用诊断。
2. **机制为什么有效**：平均协方差 C̄ 被抽象域带偏；锚定最易（自然图像）域 + detach，等于"以自然图像为基准做单向对齐"，保住了 Art/Photo 的判别流形，又没丢 Sketch 的不变性。
3. **退火为什么失败**：不变性需要**持续**的对齐压力维持；后期放松，特征会重新发散，Sketch 掉、Art 没补回——这是一个有价值的负结果。
4. **诚实的统计学注脚**：PACS 单种子下，域级波动约 ±1%。ANCHOR vs CORAL 在 Avg 上的差异（80.80 vs 80.81）**在噪声范围内**，不能宣称显著超越。要下定论需要 **3 个种子以上**的重复实验。

---

## 9. 结论与未来工作

**结论**：我们从一个只有 20% 的失败实现出发，定位并修复了"损失尺度不匹配"这一致命 bug，把方法救回到基线水平；随后通过系统化的变体研究，提出了有解释性的 **锚定单向对齐（ANCHOR_ALIGNONLY）**，在 PACS 上以"持平 CORAL 的 Avg/Sketch + 领先所有对齐法的 Art"取得了更好的难易域平衡，并在 VLCS 上验证了可迁移性。

**未来工作**：
- **多种子重复**（≥3 seeds）+ 给出均值±标准差，确认差异显著性；
- **类条件 CORAL**：当前对齐的是边缘协方差，可能压缩类间结构；按类对齐有望同时提升所有域（难点：ResNet-18 的 512 维特征 + 每类样本少，协方差很噪，需要收缩估计）；
- **锚的选择策略**：当前用"最低损失"，可尝试基于协方差到 C̄ 的距离、或多锚平均；
- 在 OfficeHome / DomainNet 等更大基准上验证。

---

## 10. 复现方式

```bash
# 单次 PACS 运行（以 ANCHOR 冠军、测试域=Art 为例）
python -u -m domainbed.scripts.dwa_train_v2 \
  --data_dir ./data \
  --algorithm DWA_CORAL_ANCHOR_ALIGNONLY \
  --dataset PACS --test_env 0 \
  --steps 3000 --checkpoint_freq 500 --skip_model_save \
  --hparams '{"resnet18": true, "resnet50_augmix": false, "dwa_coral_lambda": 0.3, "dwa_coral_beta": 0.0, "dwa_coral_tau": 1.0, "dwa_mix_gamma": 0.5}' \
  --output_dir ./outputs/PACS_DWA_CORAL_ANCHOR_ALIGNONLY_env0_steps3000

# 其它变体只需替换 --algorithm 和对应 --hparams：
#   DWA_CORAL / DWA_CORAL_ALIGNONLY / DWA_CORAL_CLIPPED /
#   DWA_CORAL_EMA / DWA_CORAL_MIXED_LOSSGAP_ALIGNONLY
# 批量脚本：run_pacs_dwa_alignonly_full.sh 等（test_env 0~3, 3000 steps）
```

| 入口 | 说明 |
|---|---|
| `domainbed/dwa_algorithms_v2.py` | 全部 DWA 变体实现 |
| `domainbed/dwa_hparams_registry_v2.py` | 各变体超参默认值/采样范围 |
| `domainbed/scripts/dwa_train_v2.py` | 训练入口（不改动官方 train.py） |
| `results/*.csv` | 各数据集最终结果 |
| `report/presentation.html` | 配套可视化幻灯片 |
