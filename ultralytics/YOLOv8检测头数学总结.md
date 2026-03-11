# YOLOv8 检测头数学总结（主线版）

## 1. YOLOv8 在学什么

我现在把 YOLOv8 检测头的学习主线收束成三块：

- Head 输出的数学对象是什么
- 这些输出如何 decode 成真实框
- 训练时如何分配正样本并计算损失

这三块就是后续看源码和公式的总框架。

## 2. Head 输出的数学对象

YOLOv8 与 YOLOv5 的关键差异是：

- YOLOv5 更像直接输出 `Δx, Δy, Δw, Δh, obj, cls`
- YOLOv8 对每个 anchor point 输出“分类 logits + 四边距离分布 logits”

对第 `i` 个点：

- 分类分支：`s_i ∈ R^(nc)`
- 回归分支：`z_i^(l), z_i^(t), z_i^(r), z_i^(b) ∈ R^K`

其中 `K = reg_max`，常用 `16`。  
所以每个点总输出维度是：`nc + 4K`，而不是一个直接的 `xywh`。

## 3. cell、anchor point、候选框的关系

一个 `H x W` 特征图对应 `H*W` 个 anchor points。  
每个点输出：

- 一组框回归分布
- 一组类别分数

例如 `40 x 40` 特征图对应 `1600` 个候选点，也就是 `1600` 组候选框预测。  
这里不是“一个点输出多个独立框模板修正”，而是“一个点对应一组框预测”。

## 4. DFL：分布到距离

以左边距离为例，网络先输出 logits：

`z^(l) = (z_0, ..., z_15)`

先 softmax 得到分布：

`p_k^(l) = exp(z_k) / Σ_j exp(z_j)`

再取期望得到连续距离：

`d_hat_l = Σ_{k=0}^{15} k * p_k^(l)`

其余三边同理得到：`d_hat_t, d_hat_r, d_hat_b`。  
所以 YOLOv8 是“先学分布，再取期望”，而不是直接回归 4 个标量距离。

## 5. dist2bbox：距离到框

设 anchor point 为 `a = (a_x, a_y)`，四边预测距离为：

`(d_hat_l, d_hat_t, d_hat_r, d_hat_b)`

则框坐标为：

- `x1 = a_x - d_hat_l`
- `y1 = a_y - d_hat_t`
- `x2 = a_x + d_hat_r`
- `y2 = a_y + d_hat_b`

这就是 dist2bbox 的本质：从参考点向左上右下扩展，而不是中心点宽高模板修正。

## 6. YOLOv5 vs YOLOv8 的本质差异

YOLOv5：

- 依赖 anchor box 宽高模板
- 预测偏移量并对模板修正

YOLOv8：

- 不使用 anchor box 模板
- 仍有 anchor points（网格参考点）
- 预测四边距离分布 logits + 分类 logits
- 经 DFL + dist2bbox 生成框

一句话：YOLOv8 是 anchor-free（box-free）但不是 point-free。

## 7. TAL：正样本分配逻辑

训练时 YOLOv8 使用 Task-Aligned Assigner（TAL），主流程：

- 候选筛选：优先考虑落在 GT 内部的点
- 对齐评分：`align_ij = s_ij^alpha * IoU_ij^beta`
- 每个 GT 选 top-k 点
- 冲突消解：一个点若被多个 GT 选中，最终只归一个 GT

因此：

- 一个 GT 通常对应多个正样本点
- 一个点最终最多服务一个 GT

## 8. 多 GT 的统一处理

同一张图可有多个 GT。  
TAL 会先对每个 GT 分别做候选和 top-k 选择，再把所有正样本统一汇总计算损失。

所以不是“每个 GT 单独训练一次”，而是“先分配，再统一进入总 loss”。

## 9. DFL 损失监督目标

若真实距离 `d = 3.7`，不会做 hard label 到单一 bin，而是分配到相邻两 bin：

- `t_l = floor(3.7) = 3`
- `t_r = 4`
- `w_l = 0.3`
- `w_r = 0.7`

DFL 可写作：

`L_DFL = w_l * CE(z, 3) + w_r * CE(z, 4)`

关键点：`w_l, w_r` 由 GT 决定，不可学习；可学习的是 logits `z`。

## 10. 反向传播在 DFL 中发生了什么

DFL 不是只优化一个标量距离，而是优化整条离散分布：

- softmax 后所有 bin 都有梯度
- 真实值邻近 bin（如 3 和 4）监督最强
- 远处 bin 更多被抑制

本质是“塑造分布形状”，不是“直接拉一个点值”。

## 11. batch 训练逻辑

训练是 batch 并行：

- 整个 batch 一次前向
- 每张图各自做 TAL 分配
- 全 batch 损失聚合
- 一次 backward 更新参数

不是逐图单独反传，而是批量统一优化。

## 12. 损失归一化与加权

YOLOv8 的损失不是简单按 batch size 平均，而是按 soft target 总权重归一化。  
可理解为：

- `L_box ~ Σ_j w_j * l_box(j) / Σ target_scores`
- `L_dfl ~ Σ_j w_j * l_dfl(j) / Σ target_scores`
- `L_cls ~ Σ BCE(pred_scores, target_scores) / Σ target_scores`

含义是：高质量正样本权重更大，低质量正样本影响更小。

## 13. 当前最核心的一句话

YOLOv8 检测头对每个 anchor point 输出分类 logits 和四边距离分布 logits；通过 softmax + 期望得到四个边距，再由 dist2bbox 解码成候选框；训练时用 TAL 分配正样本，并通过 box + cls + dfl 在整个 batch 上统一优化。

## 14. 你现在已打通的关键认知

你已经能明确回答：

- YOLOv8 到底输出什么
- 为什么是 point-based 而不是 anchor-box-based
- 为什么回归是 `4 x reg_max`
- softmax 在回归分布里做什么
- DFL 为什么把监督压在相邻 bin
- TAL 为什么不等于简单 IoU 匹配
- 多 GT / 多图 / batch 下如何统一训练
- loss 为什么是加权归一化而非简单平均

这套框架已经足够支撑你继续精读 `ultralytics` 源码里的 head、assigner 和 loss 实现。
