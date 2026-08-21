# Day 21 — Fine-tuning LLMs Lab Documentation

## Overview

This lab focuses on fine-tuning Large Language Models (LLMs) using Low-Rank Adaptation (LoRA) techniques. The primary objective is to demonstrate that a properly fine-tuned model can outperform a well-engineered prompt when solving a specific task - ticket classification into JSON triage fields.

## Project Structure

```
Day21-Track3-2A202601563-LuongNgocQuang/
├── README.md                 # Main project documentation
├── HARDWARE-GUIDE.md         # Hardware requirements and tier selection
├── rubric.md                 # Grading criteria and submission format
├── SIMULATION-FINDINGS.md    # Known issues and fixes
├── docs/
│   └── MEASURED-T4-2026-08-20.md  # Performance measurements
├── notebooks/                # Python notebook files (source)
├── colab/                    # Colab notebook files (generated)
├── data/                     # Dataset files
├── results/                  # Output results directory
├── adapters/                 # Trained adapter models
├── scripts/                  # Utility scripts
├── submission/               # Submission files (REPORT.md, REFLECTION.md)
└── src/                      # Source code modules
```

## Core Objectives

1. **Verify loss mask correctness** - Ensure the model only learns from the answer portion, not the question
2. **Establish fair baselines** - Measure performance before and after fine-tuning
3. **Configure LoRA properly** - Use the "without regret" approach with correct rank, learning rate, and batch size
4. **Design fair experiments** - Compare different configurations with identical budgets
5. **Evaluate and make judgments** - Use a four-group evaluation system (target, regression, format, latency)

## Hardware Tiers

| Tier | Hardware | Model | VRAM (bf16 LoRA) | Capabilities |
|------|----------|-------|------------------|--------------|
| `CPU` | No GPU | Qwen3.5-0.8B | — | NB1 + all tests |
| `LAPTOP` | GPU 8–12 GB | Qwen3.5-2B | ~5 GB | All notebooks |
| **`T4`** *(default)* | Colab Free T4 16 GB | Qwen3.5-4B | ~10 GB | All notebooks |
| `BIGGPU` | L4 / A100 / 3090+ | Qwen3.5-9B | ~22 GB | All notebooks |

## Quick Start

### Colab (Recommended)
1. Open [`colab/Lab21_RUN_ALL.ipynb`](https://colab.research.google.com/github/hieutrungdao/Day21-Track3-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb)
2. Runtime → Change runtime type → **T4 GPU**
3. Run cells 1 → 4

### Local Machine
```bash
git clone https://github.com/hieutrungdao/Day21-Track3-Finetuning-Lab.git
cd Day21-Track3-Finetuning-Lab
cp .env.example .env

# For CPU-only (NB1 + tests only)
make setup-cpu && make smoke && make nb1

# For GPU (recommended)
make setup && make smoke
make pipeline        # NB1 -> NB5 (~100-130 minutes on T4)
make verify          # Gatekeeper before submission
```

## Six Notebooks

| NB | Name | Time (T4) | GPU Required | Content |
|----|------|-----------|--------------|---------|
| **1** | `01_data_and_mask` | ~25 sec | ✗ | Chat template · mask proof · p95 → `max_length` · split seed 42 |
| **2** | `02_baselines` | ~17–23 min | ✓ | Frozen eval + measure baseline (a) and (b) before training |
| **3** | `03_train_correct` | ~15–25 min | ✓ | Configure regret-free zone; print `layer_types` of model |
| **4** | `04_misconfig_autopsy` | ~45–60 min | ✓ | 3 control runs: `attn_only` · `wrong_lr` · `qlora` |
| **5** | `05_evaluate_and_verdict` | ~21 min | ✓ | 4 groups · 3 baseline table · regression gate · score 3 control runs |
| 6 | `06_merge_and_serve` | ~10 min | ✓ | Merge + assert no drop + hot-swap adapter (bonus) |

## Task Description

The lab focuses on **ticket classification** - converting Vietnamese customer service tickets into structured JSON with 4 fields:
- `intent` (category)
- `urgency` (priority)
- `product` (service)
- `sentiment` (tone)

## Key Concepts

### Masking Strategy
- `MASK_MODE=assistant-only` (default) - Supervise only assistant responses
- `MASK_MODE=masked-think` - Skip reasoning traces in assistant responses
- `MASK_MODE=response-only` - Only supervise the final response

### LoRA Configuration
- **Correct configuration**: text-linear placement, rank 16, learning rate 1e-4
- **Attention-only**: q,v placement, matched rank (r≈283) 
- **Wrong learning rate**: 1e-5 instead of 1e-4
- **QLoRA**: 4-bit quantization

### Fair Experiment Design
All runs must:
1. Use the same number of optimizer steps (matched budget)
2. Have the same parameter budget (within 5% difference)
3. Only vary one variable at a time
4. Be evaluated using the same four-group metrics

### Technical Implementation Details

#### Mask Verification
The mask verification process in NB1 ensures that:
- The answer portion is correctly marked for supervision
- The question portion is excluded from loss calculation
- This is verified through `mask_proof.json` which should show both assertions as "green"

#### LoRA Placement Options
Different LoRA placements target different layers:
- `text-linear`: Targets linear layers in the text decoder
- `q,v`: Targets query and value projection layers in attention mechanisms
- `all-linear`: Targets all linear layers (can be too broad)

#### Parameter Budget Matching
The `matched_rank()` function ensures that:
- Different configurations use equivalent numbers of trainable parameters
- For example, `correct` (r=16) and `attn_only` (r≈283) are matched to use approximately the same number of parameters
- This prevents unfair comparisons where one configuration has significantly more or fewer parameters

#### Training Configuration Parameters
Key parameters that must remain consistent across runs:
- `max_steps`: Number of optimization steps (typically 30 for the default configuration)
- `per_device_train_batch_size`: Batch size per device (typically 1)
- `gradient_accumulation_steps`: Gradient accumulation steps (typically 16)
- `learning_rate`: Learning rate (typically 1e-4 for correct configuration)
- `warmup_ratio`: Warmup ratio (typically 0.1 or 10% of total steps)

### Example Command Line Usage
```bash
# Run full pipeline with default settings
make pipeline

# Run with reduced training time (half steps)
EPOCHS=1 make pipeline

# Run with reduced evaluation set
EVAL_LIMIT=8 make pipeline

# Force retraining of all adapters
FORCE_RETRAIN=1 make pipeline

# Retrain only one specific adapter
ONLY=qlora make pipeline
```

### Environment Variables
The following environment variables can be set in `.env`:
- `COMPUTE_TIER`: Select hardware tier (CPU, LAPTOP, T4, BIGGPU)
- `MASK_MODE`: Select masking strategy (assistant-only, masked-think, response-only)
- `EPOCHS`: Number of training epochs
- `EVAL_LIMIT`: Limit number of evaluation samples
- `FORCE_RETRAIN`: Force retraining of all adapters
- `ONLY`: Retrain only specific adapter

### Common Debugging Techniques
1. **Check mask correctness**: Run `scripts/check_mask_agreement.py` to verify the mask
2. **Monitor VRAM usage**: Use `torch.cuda.max_memory_allocated()` to track memory consumption
3. **Validate training progress**: Check `results/runs.csv` for training metrics
4. **Verify baseline scores**: Ensure `(b) baseline beats (a)` before training
5. **Check for OOM errors**: Monitor memory usage between notebook runs
## Key Concepts

### Masking Strategy
- `MASK_MODE=assistant-only` (default) - Supervise only assistant responses
- `MASK_MODE=masked-think` - Skip reasoning traces in assistant responses
- `MASK_MODE=response-only` - Only supervise the final response

### LoRA Configuration
- **Correct configuration**: text-linear placement, rank 16, learning rate 1e-4
- **Attention-only**: q,v placement, matched rank (r≈283) 
- **Wrong learning rate**: 1e-5 instead of 1e-4
- **QLoRA**: 4-bit quantization

### Fair Experiment Design
All runs must:
1. Use the same number of optimizer steps (matched budget)
2. Have the same parameter budget (within 5% difference)
3. Only vary one variable at a time
4. Be evaluated using the same four-group metrics

### Technical Implementation Details

#### Mask Verification
The mask verification process in NB1 ensures that:
- The answer portion is correctly marked for supervision
- The question portion is excluded from loss calculation
- This is verified through `mask_proof.json` which should show both assertions as "green"

#### LoRA Placement Options
Different LoRA placements target different layers:
- `text-linear`: Targets linear layers in the text decoder
- `q,v`: Targets query and value projection layers in attention mechanisms
- `all-linear`: Targets all linear layers (can be too broad)

#### Parameter Budget Matching
The `matched_rank()` function ensures that:
- Different configurations use equivalent numbers of trainable parameters
- For example, `correct` (r=16) and `attn_only` (r≈283) are matched to use approximately the same number of parameters
- This prevents unfair comparisons where one configuration has significantly more or fewer parameters

#### Training Configuration Parameters
Key parameters that must remain consistent across runs:
- `max_steps`: Number of optimization steps (typically 30 for the default configuration)
- `per_device_train_batch_size`: Batch size per device (typically 1)
- `gradient_accumulation_steps`: Gradient accumulation steps (typically 16)
- `learning_rate`: Learning rate (typically 1e-4 for correct configuration)
- `warmup_ratio`: Warmup ratio (typically 0.1 or 10% of total steps)

#### Example Command Line Usage
```bash
# Run full pipeline with default settings
make pipeline

# Run with reduced training time (half steps)
EPOCHS=1 make pipeline

# Run with reduced evaluation set
EVAL_LIMIT=8 make pipeline

# Force retraining of all adapters
FORCE_RETRAIN=1 make pipeline

# Retrain only one specific adapter
ONLY=qlora make pipeline
```

### Environment Variables
The following environment variables can be set in `.env`:
- `COMPUTE_TIER`: Select hardware tier (CPU, LAPTOP, T4, BIGGPU)
- `MASK_MODE`: Select masking strategy (assistant-only, masked-think, response-only)
- `EPOCHS`: Number of training epochs
- `EVAL_LIMIT`: Limit number of evaluation samples
- `FORCE_RETRAIN`: Force retraining of all adapters
- `ONLY`: Retrain only specific adapter

### Common Debugging Techniques
1. **Check mask correctness**: Run `scripts/check_mask_agreement.py` to verify the mask
2. **Monitor VRAM usage**: Use `torch.cuda.max_memory_allocated()` to track memory consumption
3. **Validate training progress**: Check `results/runs.csv` for training metrics
4. **Verify baseline scores**: Ensure `(b) baseline beats (a)` before training
5. **Check for OOM errors**: Monitor memory usage between notebook runs

### Performance Optimization Tips
1. **Memory management**: Call `generate.free_memory()` between runs to prevent memory leaks
2. **Batch size considerations**: The default batch size of 1 is optimal for T4 tier but may need adjustment for other hardware
3. **Gradient accumulation**: The default of 16 steps balances memory usage with training stability
4. **Learning rate tuning**: The default 1e-4 is optimal for most configurations, but `wrong_lr` (1e-5) demonstrates the importance of proper tuning
5. **Early stopping**: Monitor training loss curves to identify when training has converged

### Error Handling and Recovery
1. **Colab session recovery**: If a Colab session disconnects, reload the tab and restart from where it left off
2. **Adapter resumption**: Existing adapters in `adapters/` are automatically skipped during re-runs
3. **Force retraining**: Use `FORCE_RETRAIN=1` to rebuild all adapters from scratch
4. **Selective retraining**: Use `ONLY=<adapter_name>` to retrain only specific adapters
5. **Memory cleanup**: Use `make clean` to remove generated artifacts while preserving the original data

### Data and Evaluation Details
1. **Dataset structure**: The lab uses 250 customer service tickets with JSON triage fields
2. **Evaluation metrics**: All four groups (target, regression, format, latency) are scored consistently
3. **Checksum validation**: Evaluation sets are protected by checksums to prevent tampering
4. **Prompt consistency**: The optimized prompt (b) is significantly more effective than the naive prompt (a)
5. **Sample size control**: `EVAL_LIMIT` parameter allows for quick iteration with smaller sample sets

### Advanced Features
1. **Adapter merging**: NB6 demonstrates how to merge adapters and verify no performance degradation
2. **Hot-swapping**: The system supports swapping between different adapters without re-loading the base model
3. **Quantization testing**: QLoRA configuration shows the trade-off between VRAM savings and accuracy
4. **Misconfiguration analysis**: NB4 provides detailed analysis of common mistakes and their consequences
5. **Performance benchmarking**: The lab includes detailed timing measurements for different configurations

## Important Notes

### Common Issues
1. **Colab Session Limitations**: Free Colab only allows one GPU session at a time
2. **Precision Handling**: T4 GPUs lack bf16 support, so fp16 is used automatically
3. **Prompt Engineering**: Baseline (b) with optimized prompt is significantly more effective than naive prompt
4. **Memory Management**: Memory cleanup between runs is essential to prevent OOM errors

### Time Management
- Core pipeline (NB1-NB5): ~100-130 minutes on T4
- Shortened version: `EVAL_LIMIT=8 make pipeline` (17 minutes)
- Reduced training: `EPOCHS=1 make pipeline` (half training time)

## Submission Requirements

### Grading Criteria (100 + 15 bonus points)
1. **Pipeline correctness** (30 points)
2. **Fair experimental design** (25 points) 
3. **Evaluation quality** (25 points)
4. **Report quality** (20 points)

### Required Files
- `submission/REPORT.md` - Complete evaluation report
- `results/` directory with all JSON outputs
- `adapters/correct/` - Main fine-tuned adapter
- `notebooks/` - Cleaned notebooks

### Submission Options
1. **ZIP format** (~5-15 MB) - Standard submission
2. **GitHub + HuggingFace** (+2 bonus) - Repository + adapter link
3. **Code-only** (~500 KB) - Code + report + requirements

## Troubleshooting

### Common Errors
1. **OOM errors**: Lower tier or reduce batch size
2. **Import errors**: Ensure all dependencies are installed
3. **Colab session issues**: Reload tab after repository changes
4. **Training loss flat**: Check learning rate configuration

### Verification
Run `make verify` before submission to ensure:
- Mask proof passes
- Budget matching is correct
- Evaluation set checksums are intact
- Prompt (b) beats prompt (a)

## Development Environment Setup

### Requirements
- Python 3.10+
- PyTorch 2.1+
- Transformers 5.15+
- TRL 1.10+
- PEFT 0.20+

### Installation
```bash
# For CPU-only
make setup-cpu

# For GPU
make setup

# Verify installation
make smoke
```

## Performance Metrics

### Evaluation Groups
1. **Target**: Accuracy of each JSON field
2. **Regression**: Knowledge retention (15 questions)
3. **Format**: JSON parseability and completeness
4. **Latency**: Generation time per sample

### Key Measurements
- **Training Loss**: Indicates convergence
- **Target Score**: Primary metric for fine-tune effectiveness
- **VRAM Usage**: Resource consumption
- **Wall Clock Time**: Execution duration

## Best Practices

1. **Start with CPU**: Validate pipeline on CPU before GPU training
2. **Use Colab for training**: Free T4 GPU is sufficient for most tasks
3. **Monitor memory**: Clean up between runs to prevent OOM
4. **Follow the pipeline**: Execute notebooks in order (NB1-NB5)
5. **Document changes**: Record any modifications to default settings
6. **Verify results**: Always run `make verify` before submission

## References

- Deck: Day 21 — §10 (LoRA Without Regret), §13 (data & mask), §17 (evaluation), §18 (merge/serve)
- Research: LoRA (Hu et al. 2021), QLoRA (Dettmers et al. 2023), *LoRA Without Regret* (Thinking Machines 2025)
- TRL Documentation: `lora_without_regret`