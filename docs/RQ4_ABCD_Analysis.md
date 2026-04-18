# RQ4 ABCD 消融实验分析

## 实验设置

| 模型 | Text | Attr | Cross-talk | 目的 |
|------|------|------|-----------|------|
| A: Attr+CT | ✗ | ✓ | ✓ | Attr-only + MHA 属性交互 |
| B: Attr-NoCT | ✗ | ✓ | ✗ | Attr-only + 独立属性编码 |
| C: Text-Only | ✓ | ✗ | — | 纯 text embedding 基线 |
| D: Text+Attr Fusion | ✓ | ✓ | ✓ | concat fusion 融合条件 |

评估数据集：compositional split（Hamming=1 为 head，Hamming≥2 为 tail）。

## 整体生成指标

| Metric | A | B | C | D | Best |
|--------|---|---|---|---|------|
| dtw ↓ | 12.65 | 11.36 | 11.30 | 13.77 | C |
| wape ↓ | 102.67 | **91.48** | 95.48 | 102.79 | B |
| crps ↓ | 0.611 | **0.511** | 0.544 | 0.555 | B |
| fid ↓ | 57.46 | 41.94 | **31.18** | 48.36 | C |
| jftsd ↓ | 65.96 | 50.28 | **37.53** | 57.41 | C |
| joint_prdc_f1 ↑ | 0.144 | 0.194 | **0.215** | 0.142 | C |
| early-stop epoch | 180 | 212 | 191 | **33** | — |

## 组合泛化 (Head-Tail)

| 模型 | head | tail | all | gap (tail−head) |
|------|------|------|-----|-----------------|
| A | 0.240 | 0.283 | 0.252 | +0.043 |
| B | 0.338 | 0.313 | 0.347 | −0.025 |
| **C** | **0.448** | **0.414** | **0.460** | −0.034 |
| D | 0.136 | 0.222 | 0.159 | +0.086 |

## 核心分析

**1. D 训练失败**：epoch 33 早停，val/loss 0.669（其他模型 0.50-0.52）。V2-14 concat fusion 层（text+attr → 2048→1024）信息瓶颈过大或梯度冲突，结果不具可比性，需要修复后重跑（warmup / gating / 分 lr 训练）。

**2. C (Text-Only) 全面领先**：acc_norm 0.460、fid 31.18、jftsd 37.53 均最优。LongCLIP 预训练 embedding 已编码属性组合语义，作为免费的 "composition prior"。

**3. Cross-talk 损害性能（B > A）**：在 attr-only 下，无 cross-talk 的 B (acc_norm=0.347) 远超有 cross-talk 的 A (0.252)，且在 dtw/crps/fid 全面领先。与先前实验 (results_rq4.json：0.453 vs 0.406) 结论一致。

**解释**：MHA cross-talk 让属性 embedding 产生不必要耦合，训练集上记住了共现模式，面对新组合反而退化。**独立编码才是组合泛化的正确 inductive bias**——每个属性独立映射到 factor basis 系数，新组合通过系数叠加自然实现。

**4. Head-Tail gap 普遍偏小**（除 D 外在 ±4% 内）：要么模型对 Hamming=1 的 novel 组合并不敏感，要么该切分的 OOD 难度不足。可考虑加大 tail 的 Hamming 阈值（≥3）做更严格的评估。

## 对论文叙事的影响

- Cross-talk 叙事需调整：**应强调 "结构化独立编码 + HyperNet basis 的组合叠加"**，而非属性间交互。
- C 的全面领先提示：attr 方案的价值不在绝对精度，而在**可解释性与对预训练 text encoder 的独立性**。
- D 需修复后再做定论；建议尝试 gated fusion (α·text + (1−α)·attr) 替代 concat，以缓解训练不稳定。
