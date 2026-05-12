# Low-Rank Linear Subspace ReFT (LoReFT) for GPT-2 Paraphrase Detection

This project applies Low-Rank Representation Finetuning (LoReFT) to GPT-2 for paraphrase detection on the Quora Question Pairs (QQP) dataset. It compares LoReFT against full fine-tuning as a parameter-efficient alternative that modifies internal hidden representations rather than updating model weights directly.

## Motivation

This project is inspired by the paper:

> **ReFT: Representation Finetuning for Language Models**  
> Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, Christopher Potts  
> Stanford University, 2024. [arXiv:2404.03592](https://arxiv.org/abs/2404.03592)

The paper proposes the ReFT framework as an alternative to weight-based parameter-efficient finetuning (PEFT) methods such as LoRA. Rather than modifying model weights, ReFT trains lightweight intervention functions that modify selected hidden representations during the forward pass. The paper's strongest instantiation, **Low-rank Linear Subspace ReFT (LoReFT)**, performs edits within a low-dimensional subspace of the hidden representation and is shown to be 15×–65× more parameter-efficient than LoRA while achieving competitive or superior performance.

## LoReFT Implementation

The LoReFT intervention follows the formulation from the paper:

$$\Phi(h) = h + R^T(Wh + b - Rh)$$

where `h` is a hidden representation, `R` is a low-rank projection matrix, and `W` and `b` are learned parameters. The intervention modifies the representation only within the subspace spanned by `R`, keeping the number of trainable parameters small.

This is implemented as:

```python
class LoReFT(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.R = nn.Linear(d, r, bias=False)
        self.W = nn.Linear(r, r, bias=True)
        self.RT = nn.Linear(r, d, bias=False)

    def forward(self, h):
        Rh = self.R(h)
        Wrhb = self.W(Rh)
        return h + self.RT(Wrhb - Rh)
```

## Conformance to the Paper

Several design decisions in this project were made to align with the paper's specifications:

**Intervention placement.** The paper recommends intervening on `p` prefix and `s` suffix token positions. This project implements and experiments with prefix-only, suffix-only, and combined prefix+suffix strategies across different layers.

**Overlap handling.** The paper specifies that when a sequence is shorter than `p + s` tokens, the windows should shrink so prefix and suffix remain disjoint:
```
p ← min(p, ⌊n/2⌋)
s ← min(s, ⌈n/2⌉)
```
This is implemented in `p_first_s_last_nonpad_all_layers_no_loop_fix` in `loreft_strategies.py`.

**Freezing the backbone.** When using LoReFT, the GPT-2 backbone parameters are frozen and only the intervention parameters and the classification head are trained — consistent with the paper's parameter-efficient training objective.

**Rank as a hyperparameter.** Following the paper's findings, rank `r` is treated as a tunable hyperparameter. Experiments were run with `r = 8`, `r = 16`, and `r = 32`.

## Results

| Configuration | QQP Dev Accuracy |
|---|---|
| Initial LoReFT (last 2 layers, last token) | 0.725 |
| All-layer intervention (r = 8) | 0.760 |
| All-layer intervention (r = 16) | 0.770 |
| All-layer intervention (r = 32) | 0.776 |
| Prefix–suffix intervention (p = 4, s = 6) | 0.792 |
| LoReFT — tuned (5 epochs, lr = 1e-4) | **0.847** |
| Full fine-tuning baseline | 0.895 |

The best LoReFT configuration reaches 0.847 accuracy, narrowing the gap to full fine-tuning to just 0.033 while training only a small fraction of the parameters.

## Parameter Efficiency

The number of trainable parameters introduced by LoReFT is determined by the 
learned parameters ϕ = {R, W, b} per intervention, where:

- R ∈ R^(r×d): r × d parameters
- W ∈ R^(r×d): r × d parameters
- b ∈ R^r:     r parameters

Giving a per-intervention cost of:

    params_per_intervention = r(2d + 1)

The total across the model depends on the number of layers and positions intervened on:

    total_params = num_layers × (p + s) × r(2d + 1)

For GPT-2 (d = 768, 12 layers) with p = 4 prefix and s = 6 suffix positions 
intervened across all layers (120 interventions total):

    total_params = 12 × (4 + 6) × r(2 × 768 + 1) = 120 × r × 1537

| Rank r | Trainable Parameters | % of GPT-2 (117M) |
|---|---|---|
| 8  | ~1.48M | ~1.27% |
| 16 | ~2.95M | ~2.52% |
| 32 | ~5.90M | ~5.04% |

Despite training at most 5% of the model's parameters, the best LoReFT 
configuration achieves 0.847 QQP accuracy compared to 0.895 for full 
fine-tuning — a gap of only 0.048. This supports the paper's central claim 
that representation-level interventions can match weight-based fine-tuning 
at a fraction of the parameter cost.

## Project Structure

```
.
├── models/
│   ├── gpt2.py                  # GPT-2 model with LoReFT integration
│   ├── base_gpt.py              # Base GPT architecture
│   └── loreft_strategies.py     # LoReFT intervention strategies
├── paraphrase_detection.py      # Training and evaluation script
└── README.md
```

## Usage

```bash
# Full fine-tuning
python paraphrase_detection.py --epochs 5 --batch_size 32 --lr 2e-5 --use_gpu

# LoReFT
python paraphrase_detection.py --epochs 5 --batch_size 32 --lr 1e-4 --use_gpu
```

> **Note:** When switching between full fine-tuning and LoReFT, the training setup in `ParaphraseGPT` and the encode strategy in `loreft_strategies.py` must match. Full fine-tuning uses `full_finetuning()` with all parameters unfrozen. LoReFT uses any LoReFT strategy with the GPT-2 backbone frozen.
