# PTFG Technical Overview: Architecture, Conditioning, and Ablation Design

## 1. How PTFG Utilizes ST-PT

### 1.1 Background: ST-PT as a Structured Denoiser

ST-PT (Spatio-Temporal Probabilistic Transformer) performs inference on a factor graph via **Mean-Field Variational Inference (MFVI)**. Unlike standard Transformers that compute free-form attention, ST-PT decomposes the joint distribution over patch tokens into structured **factor potentials**:

$$p(\mathbf{Z} \mid \mathbf{X}) \propto \prod_i \phi_i^{\text{unary}}(z_i) \prod_{(i,j) \in \mathcal{E}_T} \psi_{ij}^{\text{ternary}}(z_i, z_j) \prod_k \chi_k^{\text{binary}}(\mathbf{z}_{\mathcal{N}(k)})$$

where:
- **Unary factors** $\phi_i$: per-patch self-information (analogous to FFN)
- **Ternary factors** $\psi_{ij}$: pairwise interactions along time and channel axes (analogous to attention)
- **Binary (topic) factors** $\chi_k$: global semantic structure (analogous to mixture-of-experts)

Each MFVI iteration is analogous to one Transformer block:

$$q^{(l+1)}(z_i) \propto \phi_i(z_i) \cdot \exp\left( \sum_{j \in \mathcal{N}(i)} \mathbb{E}_{q^{(l)}(z_j)}[\log \psi_{ij}] \right)$$

### 1.2 PTFG's Key Innovation: Condition-Programmable Factor Graphs

In the original ST-PT, factor matrices (U, V, W) are **fixed learned parameters**. PTFG makes them **condition-dependent**: the conditioning input dynamically generates the factor matrices per sample.

$$\mathbf{U}, \mathbf{V} = f_{\text{ternary}}(\mathbf{h}), \quad \mathbf{W} = f_{\text{binary}}(\mathbf{h})$$

where $\mathbf{h} = \text{CondEncoder}(\mathbf{c}, t)$ is the fused condition-timestep hidden vector.

This means: **every sample sees a different factor graph topology**, not just different features modulated by the same graph.

### 1.3 Architecture Walkthrough

The PTFG denoiser processes noisy input $\mathbf{x}_t \in \mathbb{R}^{B \times L \times C}$ as follows:

**Step 1: Patchify**

$$\mathbf{x}_t \xrightarrow{\text{reshape}} \mathbf{X}_{\text{patch}} \in \mathbb{R}^{B \times C \times P \times p}$$

where $P = L / p$ is the number of patches, $p$ is the patch length.

**Step 2: Self-Conditioning (V2-11)**

If enabled, concatenate the model's previous $\hat{\mathbf{x}}_0$ estimate:

$$\mathbf{X}_{\text{input}} = [\mathbf{X}_{\text{patch}} \; ; \; \hat{\mathbf{X}}_0] \in \mathbb{R}^{B \times C \times P \times 2p}$$

**Step 3: Patch Embedding + Unary Initialization**

$$\mathbf{Z}^{(0)} = \text{PatchEmbed}(\mathbf{X}_{\text{input}}) \in \mathbb{R}^{B \times C \times P \times D}$$

**Step 4: Condition Encoding**

$$\mathbf{h} = \text{MLP}(\text{CondProj}(\mathbf{c}) + \text{SinEmb}(t)) \in \mathbb{R}^{B \times H}$$

**Step 5: Dynamic Factor Generation**

The condition hidden $\mathbf{h}$ generates all factor matrices:

| Factor | Generator | Formula |
|--------|-----------|---------|
| Unary (AdaLN) | `UnaryConditioner` | $\phi = (\mathbf{Z} \cdot s + b) \cdot (0.5 + \sigma(\text{seg\_gate}))$ |
| Ternary U/V (time) | `StructuredTernaryFactorGenerator` | $\mathbf{U} = (\mathbf{W}_{\text{base}} + \sum_k \alpha_k \mathbf{B}_k) \odot \mathbf{r} \otimes \mathbf{c}$ |
| Ternary U/V (channel) | Same architecture | Same formula, separate parameters |
| Binary W (topic) | `BasisBinaryFactorGenerator` | $\mathbf{W} = (\mathbf{W}_{\text{base}} + \sum_k \beta_k \mathbf{B}_k) \cdot \mathbf{g}$ |

where $\alpha_k, \beta_k, \mathbf{r}, \mathbf{c}, \mathbf{g}$ are all produced by MLP heads from $\mathbf{h}$.

**Step 6: MFVI Iterations (L blocks)**

Each `DynamicPTBlock` performs one MFVI step:

$$\mathbf{m}_T = \text{TimeAttn}(\mathbf{Z}^{(l)}, \mathbf{U}_T, \mathbf{V}_T) \quad \text{(ternary: temporal message passing)}$$

$$\mathbf{m}_C = \text{ChanAttn}(\mathbf{Z}^{(l)}, \mathbf{U}_C, \mathbf{V}_C) \quad \text{(ternary: channel message passing)}$$

$$\mathbf{m}_G = \text{TopicMsg}(\mathbf{Z}^{(l)}, \mathbf{W}) \quad \text{(binary: global topic modeling)}$$

$$\mathbf{Z}^{(l+1)} = \sigma(\mathbf{g}) \cdot \mathbf{Z}^{(l)} + (1 - \sigma(\mathbf{g})) \cdot (\phi + \mathbf{m}_T + \mathbf{m}_C + \mathbf{m}_G)$$

where $\sigma(\mathbf{g})$ is a learned residual gate (replacing the fixed 0.5 MFVI damping).

**Step 7: Output Projection**

$$\hat{\mathbf{v}} = \text{Unpatchify}(\text{Linear}(\mathbf{Z}^{(L)})) \in \mathbb{R}^{B \times L \times C}$$

Under v-prediction: $\hat{\mathbf{v}} = \sqrt{\bar{\alpha}} \cdot \boldsymbol{\epsilon} - \sqrt{1 - \bar{\alpha}} \cdot \mathbf{x}_0$.

### 1.4 Why Factor Graphs, Not DiT-Style AdaLN?

| Aspect | DiT (AdaLN) | PTFG (Dynamic Factors) |
|--------|-------------|----------------------|
| What condition controls | Scale/shift of LayerNorm ($\gamma, \beta$) | U, V, W matrices (graph topology) |
| Structural effect | Element-wise feature modulation | Changes which tokens attend to which neighbors |
| Probabilistic semantics | None | MFVI provides inductive bias for uncertainty |
| Compositionality | Entangled single vector | Modular per-attribute pathways |

The key distinction: DiT modulates *features within a fixed computation graph*, while PTFG modulates *the computation graph itself*.

---

## 2. How PTFG Utilizes Conditions

### 2.1 Condition Fusion Pipeline

PTFG supports three conditioning modalities fused into a single vector:

$$\mathbf{c} = \text{Fuse}(\mathbf{c}_{\text{text}}, \mathbf{c}_{\text{attr}}, \mathbf{c}_{\text{label}})$$

| Modality | Encoding | Dimension |
|----------|----------|-----------|
| Text | Pre-computed CLIP/LongCLIP embedding $\mathbf{e}_{\text{cap}} \in \mathbb{R}^{1024}$ | `cond_dim` |
| Attribute | Per-field embedding + optional cross-talk (V2-13) | `cond_dim` |
| Label | Class embedding lookup | `cond_dim` |

### 2.2 Condition Injection Points

The condition enters the model through **four distinct pathways**:

**Pathway 1: Factor Generation** (primary)

$$\mathbf{h} = \text{CondEncoder}(\mathbf{c}, t) \xrightarrow{\text{MLP heads}} \{\mathbf{U}_T, \mathbf{V}_T, \mathbf{U}_C, \mathbf{V}_C, \mathbf{W}, \phi\}$$

This is the core mechanism: condition determines the factor graph.

**Pathway 2: Unary AdaLN + Segment Gate**

$$\phi(\mathbf{Z}) = (\mathbf{Z} \cdot (1 + 0.1 \cdot \tanh(s(\mathbf{h}))) + b(\mathbf{h})) \cdot (0.5 + \sigma(\text{seg}(\mathbf{h})))$$

The segment gate $\in \mathbb{R}^{P}$ provides per-patch weighting.

**Pathway 3: Per-Patch Ternary Modulation (V2-12)**

$$\mathbf{q}_U' = \mathbf{q}_U + \Delta\mathbf{q}(\mathbf{h}) \quad \text{where} \quad \Delta\mathbf{q} \in \mathbb{R}^{B \times P \times H \times R}$$

This adds a condition-dependent **per-patch, per-head bias** to the query of the time-dimension ternary factor. Different temporal positions receive different conditioning signals, enabling fine-grained segment-level control.

**Pathway 4: Cross-Attention to Pseudo-Tokens**

$$\text{ctx} = \text{CtxTokenizer}(\mathbf{c}) \in \mathbb{R}^{B \times N_{\text{ctx}} \times D}$$

$$\mathbf{Z}' = \mathbf{Z} + \text{CrossAttn}(\mathbf{Z}, \text{ctx}, \text{ctx})$$

The global condition is expanded into $N_{\text{ctx}}$ pseudo-tokens that patch tokens selectively attend to.

### 2.3 Attribute Conditioning with Cross-Talk (V2-13)

For discrete attributes (e.g., trend, volatility, periodicity, shape in synth-m):

$$\mathbf{e}_k = \text{Embed}_k(a_k) \in \mathbb{R}^{d_{\text{attr}}}, \quad k = 1, \ldots, K$$

Without cross-talk:

$$\mathbf{c}_{\text{attr}} = \text{MLP}([\mathbf{e}_1; \mathbf{e}_2; \ldots; \mathbf{e}_K])$$

With cross-talk (V2-13):

$$[\mathbf{e}_1'; \ldots; \mathbf{e}_K'] = \text{MHA}([\mathbf{e}_1; \ldots; \mathbf{e}_K]) + [\mathbf{e}_1; \ldots; \mathbf{e}_K]$$

$$\mathbf{c}_{\text{attr}} = \text{MLP}([\mathbf{e}_1'; \mathbf{e}_2'; \ldots; \mathbf{e}_K'])$$

The MHA allows attribute embeddings to **interact before** being projected into the condition vector, enabling synergistic effects (e.g., "high volatility" means different things depending on "uptrend" vs "downtrend").

---

## 3. RQ3: Fine-Grained Segment-Level Control

### 3.1 Research Question

> Can the model control local temporal patterns (e.g., which segment contains a shapelet) rather than just global properties?

### 3.2 Mechanism Under Test: Per-Patch Condition Modulation (V2-12)

The `PatchConditionModulator` generates a per-patch query bias:

$$\Delta\mathbf{q} = \text{MLP}(\mathbf{h}) \in \mathbb{R}^{B \times P \times H \times R}$$

This bias is added to the query in the time-dimension ternary factor attention:

$$\text{Attn}(i, j) = \frac{(\mathbf{q}_{U,i} + \Delta\mathbf{q}_i)^T \mathbf{q}_{V,j}}{d_{\text{head}}}$$

**Hypothesis**: Without this modulation, all patches receive the same condition signal, limiting the model's ability to place patterns at specific temporal positions.

### 3.3 Ablation Design

| Config | `patch_cond_modulate` | Purpose |
|--------|-----------------------|---------|
| `ptfg_best` | `true` | Full model |
| `ptfg_no_patch_mod` | `false` | Ablation: remove per-patch modulation |

### 3.4 Evaluation (Planned)

Segment-level classifier accuracy on generated samples:
1. Train a segment presence classifier on real data
2. Generate samples with specific segment-level conditions
3. Measure whether the generated patterns appear in the correct segments

---

## 4. RQ4: Compositional Generalization

### 4.1 Research Question

> Can models generalize to novel attribute combinations where multiple attribute values differ from those observed during training?

### 4.2 Mechanism Under Test: Inter-Attribute Cross-Talk (V2-13)

**Hypothesis**: Cross-talk enables attributes to form synergistic representations, improving generalization to unseen combinations. Without it, each attribute is encoded independently, and the model relies on the downstream MLP to learn interactions -- which may not generalize to novel combinations.

### 4.3 Experimental Design

#### 4.3.1 Structured Compositional Split

Standard random train/test splits of synth-m cover all 128 attribute combinations, making OOD evaluation impossible. We construct a **structured holdout**:

- Define "novel values": `attr0=3` (trend), `attr2=3` (periodicity), `attr3=3` (shape)
- **Training set**: samples where NONE of these novel values appear ($3 \times 2 \times 3 \times 3 = 54$ combos, ~42%)
- **Test set**: samples with AT LEAST ONE novel value (74 combos, ~58%)

This creates a natural Hamming distance gradient:

| Novel Attrs | Hamming Dist | Combo Count | Role |
|-------------|-------------|-------------|------|
| 0 | 0 | 54 | Train |
| 1 | 1 | 54 | Head (near-distribution) |
| 2 | 2 | 18 | Tail (OOD) |
| 3 | 3 | 2 | Tail (far OOD) |

#### 4.3.2 Ablation Configs

| Config | `attr_cross_talk` | `attribute.enabled` |
|--------|-------------------|---------------------|
| `ptfg_synth-m-compo_best` | `true` | `true` |
| `ptfg_synth-m-compo_no_cross_talk` | `false` | `true` |

#### 4.3.3 Evaluation Metric (Aligned with ConTSG Paper)

**k-NN Average Hamming Distance** for Head-Tail split:

$$d_{\text{knn}}(\mathbf{c}_{\text{test}}) = \frac{1}{k} \sum_{\mathbf{c} \in \text{KNN}_k(\mathbf{c}_{\text{test}})} \text{HD}(\mathbf{c}_{\text{test}}, \mathbf{c})$$

where $\text{HD}(\mathbf{c}_1, \mathbf{c}_2) = \sum_{j=1}^{M} \mathbb{1}[c_{1,j} \neq c_{2,j}]$.

**CTTP Retrieval Accuracy** (not classifier-based):

$$\text{Acc}_{\text{gen}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\arg\max_j \text{sim}(\mathbf{e}^{\text{ts}}_i, \mathbf{e}^{\text{text}}_j) = i\right]$$

**Normalized Accuracy**:

$$\text{Acc}_{\text{norm}} = \frac{\text{Acc}_{\text{gen}}}{\text{Acc}_{\text{ref}}}$$

**Head-Tail Gap** (OOD sensitivity):

$$\Delta = \text{Acc}_{\text{norm}}^{\text{tail}} - \text{Acc}_{\text{norm}}^{\text{head}}$$

Less negative $\Delta$ = better compositional generalization.

### 4.4 Expected Outcome

- **PTFG-best (cross-talk ON)**: Smaller Head-Tail gap, indicating that cross-talk helps maintain condition adherence on novel combinations
- **PTFG-no-cross-talk**: Larger gap, confirming that independent attribute encoding fails to generalize compositionally

---

## 5. Summary of V2 Design Items

| ID | Feature | Purpose | RQ |
|----|---------|---------|-----|
| V2-7 | Dual RoPE (time + channel) | Positional awareness for shape fidelity | - |
| V2-8 | Regularizer $1/d_{\text{head}}$ | Match original PT mean-field scaling | - |
| V2-9 | v-prediction | Numerical stability across noise levels | - |
| V2-10 | Spectral loss | Preserve autocorrelation structure (ACD) | - |
| V2-11 | Self-conditioning | Improved sample quality | - |
| V2-12 | Per-patch condition modulation | **Fine-grained segment control** | **RQ3** |
| V2-13 | Inter-attribute cross-talk | **Compositional generalization** | **RQ4** |
