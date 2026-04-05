# PTFG 技术概览：架构、条件机制与消融实验设计

## 1. PTFG 如何利用 ST-PT

### 1.1 背景：ST-PT 作为结构化去噪器

ST-PT（时空概率 Transformer）通过**均场变分推断（MFVI）**在因子图上进行推断。与标准 Transformer 的自由形式注意力不同，ST-PT 将 patch token 的联合分布分解为结构化的**因子势函数**：

$$p(\mathbf{Z} \mid \mathbf{X}) \propto \prod_i \phi_i^{\text{unary}}(z_i) \prod_{(i,j) \in \mathcal{E}_T} \psi_{ij}^{\text{ternary}}(z_i, z_j) \prod_k \chi_k^{\text{binary}}(\mathbf{z}_{\mathcal{N}(k)})$$

其中：
- **一元因子** $\phi_i$：每个 patch 的自身信息（类比 FFN）
- **三元因子** $\psi_{ij}$：沿时间轴和通道轴的成对交互（类比 attention）
- **二元因子（主题）** $\chi_k$：全局语义结构（类比 mixture-of-experts）

每次 MFVI 迭代对应一个 Transformer block：

$$q^{(l+1)}(z_i) \propto \phi_i(z_i) \cdot \exp\left( \sum_{j \in \mathcal{N}(i)} \mathbb{E}_{q^{(l)}(z_j)}[\log \psi_{ij}] \right)$$

### 1.2 PTFG 的核心创新：条件可编程因子图

在原始 ST-PT 中，因子矩阵（U, V, W）是**固定的学习参数**。PTFG 将其变为**条件相关的**：条件输入动态生成每个样本的因子矩阵。

$$\mathbf{U}, \mathbf{V} = f_{\text{ternary}}(\mathbf{h}), \quad \mathbf{W} = f_{\text{binary}}(\mathbf{h})$$

其中 $\mathbf{h} = \text{CondEncoder}(\mathbf{c}, t)$ 是条件与时间步融合后的隐藏向量。

这意味着：**每个样本看到的是不同的因子图拓扑**，而不仅仅是在同一图上做特征调制。

### 1.3 架构详解

PTFG 去噪器处理带噪输入 $\mathbf{x}_t \in \mathbb{R}^{B \times L \times C}$ 的流程如下：

**步骤 1：Patchify（分块）**

$$\mathbf{x}_t \xrightarrow{\text{reshape}} \mathbf{X}_{\text{patch}} \in \mathbb{R}^{B \times C \times P \times p}$$

其中 $P = L / p$ 是 patch 数量，$p$ 是 patch 长度。

**步骤 2：自条件化（V2-11）**

若启用，将模型上一步的 $\hat{\mathbf{x}}_0$ 估计拼接：

$$\mathbf{X}_{\text{input}} = [\mathbf{X}_{\text{patch}} \; ; \; \hat{\mathbf{X}}_0] \in \mathbb{R}^{B \times C \times P \times 2p}$$

**步骤 3：Patch Embedding + 一元势初始化**

$$\mathbf{Z}^{(0)} = \text{PatchEmbed}(\mathbf{X}_{\text{input}}) \in \mathbb{R}^{B \times C \times P \times D}$$

**步骤 4：条件编码**

$$\mathbf{h} = \text{MLP}(\text{CondProj}(\mathbf{c}) + \text{SinEmb}(t)) \in \mathbb{R}^{B \times H}$$

**步骤 5：动态因子生成**

条件隐藏向量 $\mathbf{h}$ 生成所有因子矩阵：

| 因子 | 生成器 | 公式 |
|------|--------|------|
| 一元（AdaLN） | `UnaryConditioner` | $\phi = (\mathbf{Z} \cdot s + b) \cdot (0.5 + \sigma(\text{seg\_gate}))$ |
| 三元 U/V（时间） | `StructuredTernaryFactorGenerator` | $\mathbf{U} = (\mathbf{W}_{\text{base}} + \sum_k \alpha_k \mathbf{B}_k) \odot \mathbf{r} \otimes \mathbf{c}$ |
| 三元 U/V（通道） | 同上架构 | 同上公式，独立参数 |
| 二元 W（主题） | `BasisBinaryFactorGenerator` | $\mathbf{W} = (\mathbf{W}_{\text{base}} + \sum_k \beta_k \mathbf{B}_k) \cdot \mathbf{g}$ |

其中 $\alpha_k, \beta_k, \mathbf{r}, \mathbf{c}, \mathbf{g}$ 均由 $\mathbf{h}$ 经 MLP 头生成。

**步骤 6：MFVI 迭代（L 个 block）**

每个 `DynamicPTBlock` 执行一步 MFVI：

$$\mathbf{m}_T = \text{TimeAttn}(\mathbf{Z}^{(l)}, \mathbf{U}_T, \mathbf{V}_T) \quad \text{（三元：时间维消息传递）}$$

$$\mathbf{m}_C = \text{ChanAttn}(\mathbf{Z}^{(l)}, \mathbf{U}_C, \mathbf{V}_C) \quad \text{（三元：通道维消息传递）}$$

$$\mathbf{m}_G = \text{TopicMsg}(\mathbf{Z}^{(l)}, \mathbf{W}) \quad \text{（二元：全局主题建模）}$$

$$\mathbf{Z}^{(l+1)} = \sigma(\mathbf{g}) \cdot \mathbf{Z}^{(l)} + (1 - \sigma(\mathbf{g})) \cdot (\phi + \mathbf{m}_T + \mathbf{m}_C + \mathbf{m}_G)$$

其中 $\sigma(\mathbf{g})$ 是学习的残差门控（替代固定的 0.5 MFVI 阻尼系数）。

**步骤 7：输出投影**

$$\hat{\mathbf{v}} = \text{Unpatchify}(\text{Linear}(\mathbf{Z}^{(L)})) \in \mathbb{R}^{B \times L \times C}$$

在 v-prediction 下：$\hat{\mathbf{v}} = \sqrt{\bar{\alpha}} \cdot \boldsymbol{\epsilon} - \sqrt{1 - \bar{\alpha}} \cdot \mathbf{x}_0$。

### 1.4 为什么用因子图而不是 DiT 风格的 AdaLN？

| 维度 | DiT（AdaLN） | PTFG（动态因子） |
|------|-------------|----------------|
| 条件控制什么 | LayerNorm 的缩放/偏移（$\gamma, \beta$） | U, V, W 矩阵（图拓扑） |
| 结构效果 | 逐元素特征调制 | 改变 token 之间的注意力连接模式 |
| 概率语义 | 无 | MFVI 提供不确定性的归纳偏置 |
| 组合性 | 纠缠的单一向量 | 模块化的逐属性路径 |

核心区别：DiT 在*固定计算图上调制特征*，而 PTFG *调制计算图本身*。

---

## 2. PTFG 如何利用条件

### 2.1 条件融合管线

PTFG 支持三种条件模态融合为单一向量：

$$\mathbf{c} = \text{Fuse}(\mathbf{c}_{\text{text}}, \mathbf{c}_{\text{attr}}, \mathbf{c}_{\text{label}})$$

| 模态 | 编码方式 | 维度 |
|------|---------|------|
| 文本 | 预计算 CLIP/LongCLIP 嵌入 $\mathbf{e}_{\text{cap}} \in \mathbb{R}^{1024}$ | `cond_dim` |
| 属性 | 逐字段 embedding + 可选 cross-talk（V2-13） | `cond_dim` |
| 标签 | 类别 embedding 查表 | `cond_dim` |

### 2.2 条件注入点

条件通过**四条独立路径**进入模型：

**路径 1：因子生成**（核心路径）

$$\mathbf{h} = \text{CondEncoder}(\mathbf{c}, t) \xrightarrow{\text{MLP heads}} \{\mathbf{U}_T, \mathbf{V}_T, \mathbf{U}_C, \mathbf{V}_C, \mathbf{W}, \phi\}$$

这是核心机制：条件决定因子图。

**路径 2：一元 AdaLN + 段门控**

$$\phi(\mathbf{Z}) = (\mathbf{Z} \cdot (1 + 0.1 \cdot \tanh(s(\mathbf{h}))) + b(\mathbf{h})) \cdot (0.5 + \sigma(\text{seg}(\mathbf{h})))$$

段门控 $\in \mathbb{R}^{P}$ 提供逐 patch 的权重。

**路径 3：逐 Patch 三元因子调制（V2-12）**

$$\mathbf{q}_U' = \mathbf{q}_U + \Delta\mathbf{q}(\mathbf{h}) \quad \text{其中} \quad \Delta\mathbf{q} \in \mathbb{R}^{B \times P \times H \times R}$$

在时间维三元因子的 query 上添加条件相关的**逐 patch、逐 head 偏置**。不同时间位置接收不同的条件信号，实现细粒度的段级控制。

**路径 4：伪 Token 交叉注意力**

$$\text{ctx} = \text{CtxTokenizer}(\mathbf{c}) \in \mathbb{R}^{B \times N_{\text{ctx}} \times D}$$

$$\mathbf{Z}' = \mathbf{Z} + \text{CrossAttn}(\mathbf{Z}, \text{ctx}, \text{ctx})$$

全局条件被展开为 $N_{\text{ctx}}$ 个伪 token，patch token 可以选择性地关注它们。

### 2.3 属性条件与交叉对话（V2-13）

对于离散属性（如 synth-m 中的趋势、波动率、周期性、形状）：

$$\mathbf{e}_k = \text{Embed}_k(a_k) \in \mathbb{R}^{d_{\text{attr}}}, \quad k = 1, \ldots, K$$

无 cross-talk 时：

$$\mathbf{c}_{\text{attr}} = \text{MLP}([\mathbf{e}_1; \mathbf{e}_2; \ldots; \mathbf{e}_K])$$

有 cross-talk（V2-13）时：

$$[\mathbf{e}_1'; \ldots; \mathbf{e}_K'] = \text{MHA}([\mathbf{e}_1; \ldots; \mathbf{e}_K]) + [\mathbf{e}_1; \ldots; \mathbf{e}_K]$$

$$\mathbf{c}_{\text{attr}} = \text{MLP}([\mathbf{e}_1'; \mathbf{e}_2'; \ldots; \mathbf{e}_K'])$$

MHA 允许属性嵌入在投影为条件向量**之前进行交互**，实现协同效应（例如，"高波动" 在 "上升趋势" 和 "下降趋势" 下含义不同）。

---

## 3. RQ3：细粒度段级控制

### 3.1 研究问题

> 模型能否控制局部时间模式（如 shapelet 出现在哪个段），而不仅是全局属性？

### 3.2 测试机制：逐 Patch 条件调制（V2-12）

`PatchConditionModulator` 生成逐 patch 的 query 偏置：

$$\Delta\mathbf{q} = \text{MLP}(\mathbf{h}) \in \mathbb{R}^{B \times P \times H \times R}$$

该偏置添加到时间维三元因子注意力的 query 上：

$$\text{Attn}(i, j) = \frac{(\mathbf{q}_{U,i} + \Delta\mathbf{q}_i)^T \mathbf{q}_{V,j}}{d_{\text{head}}}$$

**假设**：没有此调制时，所有 patch 接收相同的条件信号，限制了模型在特定时间位置放置模式的能力。

### 3.3 消融设计

| 配置 | `patch_cond_modulate` | 目的 |
|------|-----------------------|------|
| `ptfg_best` | `true` | 完整模型 |
| `ptfg_no_patch_mod` | `false` | 消融：移除逐 patch 调制 |

### 3.4 评估方案（计划中）

在生成样本上的段级分类器准确率：
1. 在真实数据上训练段存在性分类器
2. 用特定段级条件生成样本
3. 测量生成的模式是否出现在正确的段中

---

## 4. RQ4：组合泛化

### 4.1 研究问题

> 模型能否泛化到训练中未见过的属性组合，其中多个属性值与训练分布不同？

### 4.2 测试机制：属性间交叉对话（V2-13）

**假设**：Cross-talk 使属性形成协同表示，提高对未见组合的泛化。没有它时，每个属性独立编码，模型依赖下游 MLP 学习交互——这可能无法泛化到新组合。

### 4.3 实验设计

#### 4.3.1 结构化组合分割

synth-m 的标准随机 train/test 分割覆盖了全部 128 种属性组合，使 OOD 评估不可能。我们构建**结构化 holdout**：

- 定义"新颖值"：`attr0=3`（趋势）、`attr2=3`（周期性）、`attr3=3`（形状）
- **训练集**：所有新颖值都不出现的样本（$3 \times 2 \times 3 \times 3 = 54$ 种组合，约 42%）
- **测试集**：至少包含一个新颖值的样本（74 种组合，约 58%）

这创造了天然的 Hamming 距离梯度：

| 新颖属性数 | Hamming 距离 | 组合数 | 角色 |
|-----------|-------------|--------|------|
| 0 | 0 | 54 | 训练集 |
| 1 | 1 | 54 | Head（近分布） |
| 2 | 2 | 18 | Tail（OOD） |
| 3 | 3 | 2 | Tail（远 OOD） |

#### 4.3.2 消融配置

| 配置 | `attr_cross_talk` | `attribute.enabled` |
|------|-------------------|---------------------|
| `ptfg_synth-m-compo_best` | `true` | `true` |
| `ptfg_synth-m-compo_no_cross_talk` | `false` | `true` |

#### 4.3.3 评估指标（与 ConTSG 论文对齐）

**k-NN 平均 Hamming 距离**用于 Head-Tail 分割：

$$d_{\text{knn}}(\mathbf{c}_{\text{test}}) = \frac{1}{k} \sum_{\mathbf{c} \in \text{KNN}_k(\mathbf{c}_{\text{test}})} \text{HD}(\mathbf{c}_{\text{test}}, \mathbf{c})$$

其中 $\text{HD}(\mathbf{c}_1, \mathbf{c}_2) = \sum_{j=1}^{M} \mathbb{1}[c_{1,j} \neq c_{2,j}]$。

**CTTP 检索准确率**（非分类器方式）：

$$\text{Acc}_{\text{gen}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\arg\max_j \text{sim}(\mathbf{e}^{\text{ts}}_i, \mathbf{e}^{\text{text}}_j) = i\right]$$

**归一化准确率**：

$$\text{Acc}_{\text{norm}} = \frac{\text{Acc}_{\text{gen}}}{\text{Acc}_{\text{ref}}}$$

**Head-Tail 差距**（OOD 敏感度）：

$$\Delta = \text{Acc}_{\text{norm}}^{\text{tail}} - \text{Acc}_{\text{norm}}^{\text{head}}$$

$\Delta$ 越接近 0（越不负）= 组合泛化越好。

### 4.4 预期结果

- **PTFG-best（cross-talk 开启）**：更小的 Head-Tail 差距，表明 cross-talk 帮助在新组合上维持条件遵循
- **PTFG-no-cross-talk**：更大的差距，确认独立属性编码无法实现组合泛化

---

## 5. V2 设计项汇总

| 编号 | 特性 | 目的 | 对应 RQ |
|------|------|------|---------|
| V2-7 | 双 RoPE（时间+通道） | 位置感知以保证形状保真度 | - |
| V2-8 | 正则化 $1/d_{\text{head}}$ | 匹配原始 PT 均场缩放 | - |
| V2-9 | v-prediction | 全噪声水平下的数值稳定性 | - |
| V2-10 | 频谱损失 | 保持自相关结构（ACD） | - |
| V2-11 | 自条件化 | 提高样本质量 | - |
| V2-12 | 逐 Patch 条件调制 | **细粒度段级控制** | **RQ3** |
| V2-13 | 属性间交叉对话 | **组合泛化** | **RQ4** |
