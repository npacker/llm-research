# Dynamic Temperature for Generative Replay

---

## Area 1: Token-Level vs. Sequence-Level Temperature

### Research Question

**Does token-level dynamic temperature (EDT-style) outperform sequence-level temperature assignment for generative replay, and under what conditions?**

### Hypothesis

Token-level adjustment provides finer control but may introduce instability in long sequences. Sequence-level temperature (one temperature per generated sample) may provide better coherence while sacrificing some diversity.

### Study Methodology

#### Experimental Design

```
┌─────────────────────────────────────────────────────────────────┐
│            TEMPERATURE GRANULARITY COMPARISON                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Condition A: Token-Level EDT                                   │
│  ├─ Calculate entropy at each token position                    │
│  ├─ Adjust temperature per-token using EDT formula              │
│  └─ T_token = T₀ × N^(θ/entropy_at_position)                    │
│                                                                 │
│  Condition B: Sequence-Level Fixed                              │
│  ├─ Calculate average entropy over first 20% of sequence        │
│  ├─ Assign one temperature for entire generation                │
│  └─ T_sequence = T₀ × N^(θ/avg_entropy)                         │
│                                                                 │
│  Condition C: Hybrid (Stage-Based)                              │
│  ├─ High temp for first 30% tokens (exploration phase)          │
│  ├─ Medium temp for middle 40% (transition)                     │
│  └─ Low temp for final 30% (commitment phase)                   │
│                                                                 │
│  Condition D: Fixed Temperature Baseline                        │
│  └─ Standard fixed T = 0.7, 0.8, 0.9                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> **EDT temperature formula — symbol definitions.** In `T = T₀ × N^(θ/entropy)`:
> - `T₀` — base temperature.
> - `entropy` — Shannon entropy of the next-token distribution at the current position (in nats; use a small ε floor to avoid division by zero).
> - `N` — base of the exponential adjustment (hyperparameter, `N > 0`).
> - `θ` — exponent scaling factor (hyperparameter) controlling how strongly entropy modulates temperature.
>
> Verify the sign/direction of the adjustment (does higher entropy raise or lower `T`?) and the
> default values of `N`, `θ` against the EDT source (Zhang et al., 2024, *Entropy-based Dynamic
> Temperature Sampling*) before implementation — the intended behavior is to damp temperature
> when the model is uncertain and raise it when confident.

#### Implementation Protocol

1. **Model Selection:**
   - Use LLaMA-2-7B or Qwen-2.5-7B (open weights, well-documented)
   - Fine-tune on domain-specific corpus first (e.g., instruction data)

2. **Generation Setup:**
   - Generate 10,000 synthetic samples per condition
   - Prefix-only generation (structural markers only)
   - Max tokens: 256-512 (varied by task)

3. **Data Collection:**
   - Store full generation trajectories (all token probabilities)
   - Log entropy at each position
   - Record final temperature values used

### Benchmarking Approach

| Metric | Measurement | Target |
|--------|-------------|--------|
| **Perplexity** | Language model perplexity on generated text | Lower = better coherence |
| **Self-BLEU** | N-gram overlap between generated samples | Lower = more diverse |
| **Distinct-n** | Ratio of unique n-grams to total n-grams | Higher = more diverse |
| **Entropy Variance** | Standard deviation of token entropies within sequence | Measures temperature fluctuation |
| **Coherence Score** | LLM-as-judge rating (1-5 scale) | Higher = better quality |
| **Task Accuracy** | If applicable (e.g., math problems) | Correctness rate |

### Expected Challenges

1. **Computational Overhead:** Token-level EDT requires entropy calculation at each step (~2-3x inference time)
2. **Temperature Instability:** Extreme entropy values may cause temperature spikes
3. **Evaluation Complexity:** Need both local (token) and global (sequence) metrics

### Potential Impact

- **High:** Could establish best practices for temperature granularity in synthetic data generation
- **Publishable:** Novel comparison with clear experimental methodology
- **Practical:** Direct implementation guidance for practitioners

### Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Literature Review | 2 weeks | Annotated bibliography |
| Implementation | 3 weeks | Working codebase for all conditions |
| Data Generation | 2 weeks | 40,000+ synthetic samples |
| Evaluation | 3 weeks | Metrics, statistical analysis |
| Writing | 2 weeks | Paper draft |
| **Total** | **12 weeks** | **Complete study** |

---

## Area 2: Curriculum-Based Temperature Scheduling

### Research Question

**Can temperature be scheduled across training epochs to balance early diversity with late-stage fidelity, and does this prevent model collapse in recursive training?**

### Hypothesis

A curriculum that starts with high temperature (exploration) and gradually decreases (exploitation) will produce better replay data than fixed temperature, especially over multiple generations of recursive training.

### Study Methodology

#### Scheduling Strategies to Test

*Pseudocode (illustrative). Assumes `from math import cos, pi`; `threshold` and
`calculate_diversity_decrease` are placeholders to be defined at implementation time.*

```python
# Curriculum scheduling functions for temperature

def cosine_decay_schedule(epoch, total_epochs, T_max, T_min):
    """Smooth decay from high to low temperature"""
    return T_min + (T_max - T_min) * (1 + cos(pi * epoch / total_epochs)) / 2

def linear_decay_schedule(epoch, total_epochs, T_max, T_min):
    """Linear decrease"""
    return T_max - (T_max - T_min) * (epoch / total_epochs)

def phase_based_schedule(epoch, total_epochs):
    """Three-phase: explore, balance, commit"""
    phase = epoch / total_epochs
    if phase < 0.3:
        return 0.9  # High diversity
    elif phase < 0.7:
        return 0.7  # Balanced
    else:
        return 0.5  # High fidelity

def entropy_adaptive_schedule(generation_stats, T_base):
    """Adjust based on observed diversity metrics"""
    diversity_drop = calculate_diversity_decrease(generation_stats)
    if diversity_drop > threshold:
        return T_base * 1.2  # Increase temp to restore diversity
    return T_base
```

#### Experimental Design

**Recursive Training Loop:**

```
Generation 0: Real data (baseline)
    ↓
Generation 1: Synthetic data with T_schedule_A
    ↓
Generation 2: Synthetic data with T_schedule_A (trained on Gen 1)
    ↓
Generation 3: Synthetic data with T_schedule_A (trained on Gen 2)
    ↓
...
Generation N: Measure collapse indicators
```

**Conditions:**
- **A:** Cosine decay (T: 0.9 → 0.5 over 5 epochs)
- **B:** Linear decay (T: 0.9 → 0.5 over 5 epochs)
- **C:** Phase-based (3 distinct temperature phases)
- **D:** Entropy-adaptive (dynamic based on diversity metrics)
- **E:** Fixed temperature control (T = 0.7)

### Benchmarking Approach

#### Model Collapse Metrics

| Metric | Measurement | Collapse Threshold |
|--------|-------------|-------------------|
| **Vocabulary Shrinkage** | Type-token ratio (unique tokens / total tokens) | < 70% of baseline (Gen 0) — kept consistent with Area 5's vocabulary threshold |
| **Distribution KL-Divergence** | KL(P_real ‖ P_synthetic), token/feature distribution | > 2.0 nats |
| **Quality Degradation** | LLM judge score drop | > 20% from baseline |
| **Diversity Loss** | Self-BLEU increase | > 30% from baseline |
| **Task Performance** | Downstream model accuracy | > 15% drop |

#### Additional Metrics

- **Convergence Speed:** How quickly does temperature stabilize?
- **Optimal Switch Point:** When should exploration → exploitation transition occur?
- **Generation Depth Limit:** How many recursive generations before collapse?

### Expected Challenges

1. **Long Experimental Timeline:** Recursive training requires multiple model training cycles
2. **Compute Resources:** Training models for each generation is expensive
3. **Confounding Variables:** Hard to isolate temperature effects from other factors

### Mitigation Strategies

- Use **LoRA fine-tuning** to reduce compute costs
- Start with **smaller models** (1-3B parameters) for pilot studies
- Use **proxy metrics** (diversity measures) instead of full model training for early phases

### Potential Impact

- **Very High:** Model collapse prevention is a critical open problem
- **Novel:** First systematic study of temperature scheduling for collapse prevention
- **Citable:** Addresses Nature (2024) and NeurIPS findings on model collapse

### Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Pilot Study (small models) | 4 weeks | Feasibility confirmation |
| Full Recursive Training | 8 weeks | 5+ generations of data |
| Collapse Analysis | 3 weeks | Metrics, visualization |
| Writing | 3 weeks | Paper draft |
| **Total** | **18 weeks** | **Complete study** |

---

## Area 3: Task-Adaptive Temperature

### Research Question

**Do different task types (code, math, creative writing, factual QA) require different temperature profiles for optimal replay data generation?**

### Hypothesis

Task characteristics (precision requirements, creativity needs, structural constraints) interact with temperature settings, and adaptive per-task temperature outperforms uniform temperature across mixed task corpora.

### Study Methodology

#### Task Categories to Investigate

| Task Type | Precision Need | Creativity Need | Recommended T Range |
|-----------|---------------|-----------------|---------------------|
| **Code Generation** | Very High | Medium | 0.3 - 0.6 |
| **Math Reasoning** | Very High | Low | 0.2 - 0.5 |
| **Factual QA** | High | Low | 0.4 - 0.7 |
| **Creative Writing** | Low | Very High | 0.7 - 1.0 |
| **Instruction Following** | Medium | Medium | 0.5 - 0.8 |
| **Dialogue/Chat** | Low-Medium | High | 0.6 - 0.9 |

#### Experimental Design

**Phase 1: Task-Specific Optimization**
- For each task type, sweep temperature from 0.2 to 1.0 (step 0.1)
- Generate 1,000 samples per temperature per task
- Identify optimal temperature range for each task

**Phase 2: Adaptive Assignment**
- Build task classifier (or use task labels if available)
- Assign temperature based on detected task type
- Compare against uniform temperature baseline

**Phase 3: Fine-Grained Adaptation**
- Within-task temperature adjustment based on:
  - Sequence entropy (uncertainty)
  - Token type (code vs. natural language within same sequence)
  - Position in sequence (structure vs. content)

### Benchmarking Approach

#### Task-Specific Metrics

| Task | Primary Metric | Secondary Metric |
|------|---------------|------------------|
| **Code** | Pass@k (execution success) | Code complexity, style consistency |
| **Math** | Answer accuracy | Reasoning chain validity |
| **Factual QA** | Fact verification score | Citation accuracy |
| **Creative** | Human creativity rating | Lexical diversity |
| **Instruction** | Instruction following score | Response helpfulness |

#### Cross-Task Metrics

- **Overall Quality:** Weighted average across all tasks
- **Fairness:** Performance variance across task types (lower = more equitable)
- **Adaptation Accuracy:** How well does task detection match optimal temperature?

### Expected Challenges

1. **Task Detection:** Automatic task classification may be error-prone
2. **Metric Heterogeneity:** Different tasks require different evaluation approaches
3. **Boundary Cases:** Many samples span multiple task categories

### Potential Impact

- **High:** Practical guidance for multi-task synthetic data generation
- **Immediately Applicable:** Can be implemented in existing pipelines
- **Foundation:** Enables more sophisticated adaptive generation systems

### Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Task Corpus Preparation | 2 weeks | Labeled dataset across 6 task types |
| Temperature Sweep | 3 weeks | Optimal ranges identified |
| Adaptive System Build | 3 weeks | Working task-adaptive generator |
| Evaluation | 3 weeks | Cross-task comparison |
| Writing | 2 weeks | Paper draft |
| **Total** | **13 weeks** | **Complete study** |

---

## Area 4: Interaction with Prefix-Only Generation

### Research Question

**How does dynamic temperature interact with prefix-only generation, and does combining both techniques produce synergistic effects for distribution matching?**

### Hypothesis

Prefix-only generation provides structural distribution alignment, while dynamic temperature provides content diversity. Combined, they achieve better distribution matching than either technique alone.

### Study Methodology

#### Experimental Conditions

```
┌─────────────────────────────────────────────────────────────────┐
│         PREFIX × TEMPERATURE FACTORIAL DESIGN                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Prefix Conditions:                                             │
│  ├─ P1: No prefix (standard prompting)                          │
│  ├─ P2: Structural prefix only (format markers)                 │
│  ├─ P3: Real data snippet prefix (50% of demonstration)         │
│  └─ P4: Variable prefix length (curriculum scheduling)          │
│                                                                 │
│  Temperature Conditions:                                        │
│  ├─ T1: Fixed temperature (T = 0.7)                             │
│  ├─ T2: Token-level EDT                                         │
│  ├─ T3: Sequence-level EDT                                      │
│  └─ T4: Curriculum-scheduled temperature                        │
│                                                                 │
│  Total: 16 experimental conditions (4 × 4 factorial)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Benchmarking Approach

#### Distribution Matching Metrics

| Metric | Measurement | Interpretation |
|--------|-------------|----------------|
| **Maximum Mean Discrepancy (MMD)** | Statistical distance between real and synthetic distributions | Lower = better match |
| **Embedding Cosine Similarity** | Average similarity between real/synthetic sentence embeddings | Higher = better match |
| **Topic Distribution KL-Divergence** | KL(P_real ‖ P_synthetic) over topic model (LDA/BERTopic) | Lower = better coverage |
| **Style Embedding Distance** | Writing style representation comparison | Lower = better style match |
| **Length Distribution Match** | Histogram comparison of sequence lengths | Lower = better structural match |

#### Synergy Analysis

*Pseudocode (illustrative). The `p1_*`, `t1_*`, and `combined_*` operands are placeholders
for measured metric values populated during analysis.*

```python
# Calculate interaction effects between prefix and temperature

def calculate_synergy_score(prefix_effect, temp_effect, combined_effect):
    """
    Measure if prefix + temperature combination produces synergistic effects.
    
    Synergy = Combined - (Prefix + Temperature)
    Positive synergy = better than sum of individual effects
    Negative synergy = interference between techniques
    """
    additive_expectation = prefix_effect + temp_effect
    synergy = combined_effect - additive_expectation
    return synergy

# Apply to multiple metrics
synergy_scores = {
    'mmd': calculate_synergy_score(p1_mmd, t1_mmd, combined_mmd),
    'diversity': calculate_synergy_score(p1_div, t1_div, combined_div),
    'quality': calculate_synergy_score(p1_qual, t1_qual, combined_qual),
}
```

#### Primary Outcomes

1. **Distribution Alignment Score:** Composite metric combining MMD, KL-divergence, and embedding similarity
2. **Diversity Preservation:** Self-BLEU, Distinct-n, vocabulary coverage
3. **Downstream Performance:** Train a model on synthetic data, test on real data benchmarks

#### Statistical Analysis

- **ANOVA:** Test for significant main effects and interactions
- **Effect Sizes:** Cohen's d for practical significance
- **Bayesian Analysis:** Quantify evidence for synergy vs. additivity

### Expected Challenges

1. **Computational Cost:** 16 conditions × 10,000 samples = 160,000 generations
2. **Metric Selection:** No standardized benchmark for distribution matching quality
3. **Confounding Variables:** Prefix quality and temperature effects may be correlated

### Potential Impact

- **High:** Establishes best practices for combining two emerging techniques
- **Novel:** First factorial study of prefix × temperature interaction
- **Practical:** Clear implementation guidance for synthetic data pipelines

### Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Condition Setup | 2 weeks | All 16 generation pipelines |
| Data Generation | 4 weeks | 160,000+ synthetic samples |
| Distribution Analysis | 3 weeks | MMD, KL, embedding metrics |
| Downstream Training | 3 weeks | Model training & evaluation |
| Writing | 2 weeks | Paper draft |
| **Total** | **14 weeks** | **Complete study** |

---

## Area 5: Model Collapse Prevention

### Research Question

**Can entropy-based dynamic temperature sampling interrupt or delay model collapse in recursive synthetic data training, and what are the critical parameters for collapse prevention?**

### Hypothesis

Dynamic temperature maintains diversity in the generative process, preventing the positive feedback loop that drives model collapse. There exists an optimal temperature range that balances diversity preservation with quality maintenance.

### Background: Model Collapse Mechanisms

Based on Nature (2024) and subsequent research, model collapse occurs through:

1. **Tail Disappearance:** Low-probability tokens become increasingly rare across generations
2. **Mode Collapse:** Distribution concentrates on high-probability outputs
3. **Error Accumulation:** Small errors compound across recursive training
4. **Diversity Loss:** Self-reinforcing homogenization of outputs

```
┌─────────────────────────────────────────────────────────────────┐
│              MODEL COLLAPSE PROGRESSION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Generation 0 (Real Data):                                      │
│  ████████▓▓▓▓▒▒░░  (Full distribution, long tail)               │
│                                                                 │
│  Generation 1 (Synthetic):                                      │
│  ██████████▓▓▓▒▒░░ (Tail begins shrinking)                      │
│                                                                 │
│  Generation 2 (Synthetic):                                      │
│  ████████████▓▓▓▒░░ (Further concentration)                     │
│                                                                 │
│  Generation 3+ (Collapse):                                      │
│  ████████████████▓▓▓░ (Severe mode collapse)                    │
│                                                                 │
│  WITH DYNAMIC TEMPERATURE:                                      │
│  Maintains tail diversity through entropy-aware sampling        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Study Methodology

#### Experimental Design

**Recursive Training Protocol:**

*Pseudocode (illustrative). `generate_with_temperature_strategy`, `mix_data`, `train_model`,
and `evaluate_collapse_indicators` are helpers to be implemented.*

```python
for generation in range(10):
    
    # Generate synthetic data with temperature strategy
    synthetic_data = generate_with_temperature_strategy(
        model=current_model,
        temperature_strategy=condition,  # EDT, fixed, adaptive, etc.
        n_samples=50000
    )
    
    # Mix with real data (varying ratios)
    training_data = mix_data(
        real_data=real_corpus,
        synthetic_data=synthetic_data,
        real_ratio=real_data_percentage  # 0%, 20%, 50%, 100%
    )
    
    # Train next generation model
    next_model = train_model(training_data)
    
    # Measure collapse indicators
    metrics = evaluate_collapse_indicators(next_model)
    
    current_model = next_model
```

#### Temperature Strategies to Test

| Condition | Description | Rationale |
|-----------|-------------|-----------|
| **T1: Fixed Low (0.5)** | Conservative, high quality | Baseline for quality-focused generation |
| **T2: Fixed Medium (0.7)** | Standard practice | Common default in many pipelines |
| **T3: Fixed High (0.9)** | High diversity | Baseline for diversity-focused generation |
| **T4: Token-Level EDT** | Entropy-based per-token | Tests EDT paper approach |
| **T5: Sequence-Level EDT** | Entropy-based per-sequence | Tests simplified EDT |
| **T6: Curriculum Decay** | High→Low over generations | Tests scheduling hypothesis |
| **T7: Diversity-Adaptive** | Increase T when diversity drops | Tests feedback control |
| **T8: Hybrid (EDT + Real Mix)** | EDT with 20% real data | Tests combined prevention |

### Benchmarking Approach

#### Collapse Indicators (Primary)

| Metric | Measurement | Collapse Threshold |
|--------|-------------|-------------------|
| **Vocabulary Size** | Unique tokens in generated corpus | < 70% of Generation 0 |
| **Tail Probability Mass** | Probability of bottom 20% tokens | < 50% of Generation 0 |
| **Self-BLEU Increase** | N-gram overlap between generations | > 40% increase |
| **Perplexity on Real Held-out** | Model's perplexity on a fixed *real* held-out test set | > 30% **increase** (worse real-data fit). NB: perplexity on the model's *own* generations tends to drop as it grows overconfident — measure both and report which set. |
| **Quality Score** | LLM-as-judge rating | > 25% degradation |

#### Collapse Indicators (Secondary)

| Metric | Measurement | Significance |
|--------|-------------|--------------|
| **Embedding Space Contraction** | Variance of sentence embeddings | Measures semantic diversity |
| **Topic Concentration** | Topic model entropy | Measures thematic diversity |
| **Syntax Pattern Repetition** | Parse tree pattern frequency | Measures structural diversity |
| **Error Rate** | Factual/logical errors in output | Measures quality degradation |

#### Prevention Effectiveness Metrics

| Metric | Calculation | Target |
|--------|-------------|--------|
| **Collapse Delay** | Number of generations before threshold crossed | Higher = better |
| **Collapse Severity** | Final metric value at generation 10 | Lower = better |
| **Recovery Potential** | Can performance recover with real data injection? | Yes/No + speed |
| **Compute Efficiency** | Quality maintained per training dollar | Higher = better |

### Expected Challenges

1. **Long Timeline:** 10 generations of model training is computationally intensive
2. **Attribution:** Hard to separate temperature effects from other collapse factors
3. **Definition:** No consensus on exact collapse threshold or measurement
4. **Scale:** Small models may not exhibit same collapse patterns as large models

### Mitigation Strategies

- **Use Proxy Models:** Run pilot studies with 1B parameter models
- **Early Stopping:** Monitor collapse indicators and stop if severe degradation observed
- **Incremental Approach:** Start with 3-5 generations, extend if promising
- **Collaboration:** Partner with institutions having larger compute resources

### Theoretical Framework

```
┌─────────────────────────────────────────────────────────────────┐
│         COLLAPSE PREVENTION THEORETICAL MODEL                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Collapse Driver:                                               │
│  D(t+1) = D(t) × (1 - α) + ε  [Diversity decay]                 │
│  Where α = concentration rate, ε = noise                        │
│                                                                 │
│  EDT Intervention:                                              │
│  D(t+1) = D(t) × (1 - α + β·entropy) + ε                        │
│  Where β = temperature sensitivity parameter                    │
│                                                                 │
│  Critical Threshold:                                            │
│  If entropy > entropy_critical → diversity maintained           │
│  If entropy < entropy_critical → collapse accelerates           │
│                                                                 │
│  Research Question: What is entropy_critical for different      │
│  model sizes and task types?                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Potential Impact

- **Very High:** Model collapse is a critical threat to synthetic data pipelines
- **Policy-Relevant:** Findings could inform AI safety guidelines
- **Foundational:** Establishes theoretical understanding of collapse dynamics
- **High Citation Potential:** Addresses widely-cited Nature (2024) findings

### Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Pilot Study (3 generations) | 6 weeks | Feasibility & parameter tuning |
| Full Study (10 generations) | 12 weeks | Complete collapse trajectory |
| Analysis | 4 weeks | Statistical modeling, visualization |
| Writing | 4 weeks | Paper draft |
| **Total** | **26 weeks** | **Complete study** |

---

## Cross-Area Synthesis & Combined Research Program

### Integrated Research Design

These five areas can be studied as a **coordinated research program** rather than isolated investigations:

```
┌─────────────────────────────────────────────────────────────────┐
│            INTEGRATED RESEARCH TIMELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Months 1-3:  Area 1 (Token vs. Sequence)                       │
│               Area 3 (Task-Adaptive)                            │
│               [Parallel: foundational comparisons]              │
│                                                                  │
│  Months 4-6:  Area 4 (Prefix × Temperature Interaction)         │
│               Area 2 (Curriculum Scheduling)                    │
│               [Build on Months 1-3 findings]                    │
│                                                                  │
│  Months 7-12: Area 5 (Model Collapse Prevention)                │
│               [Integrate all findings into collapse study]      │
│                                                                  │
│  Month 13+:   Synthesis paper, open-source release              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Hardware Feasibility (single-GPU note)

This program is scoped for a **single RTX PRO 6000 Blackwell (96 GB)**. The heavier studies —
especially Area 5 (≈50k samples × 10 generations × 8 conditions, with a model trained each
generation) — are large for one GPU run end-to-end at full scale. Lead with the **pilot-first
path** the individual areas already prescribe: LoRA fine-tuning, 1–3B pilot models, fp8 KV-cache
for longer context, early stopping on collapse indicators, and `tensor_parallel_size=1`. Tag each
study with a rough single-GPU compute estimate before committing to the full-scale matrix.

### Shared Infrastructure

| Component | Purpose | Reuse Across Areas |
|-----------|---------|-------------------|
| **Generation Pipeline** | Synthetic data generation with EDT | All 5 areas |
| **Metrics Library** | Diversity, quality, distribution metrics | All 5 areas |
| **Model Training** | LoRA fine-tuning infrastructure | Areas 2, 5 |
| **Evaluation Framework** | Automated benchmarking suite | All 5 areas |
| **Data Storage** | Versioned synthetic data corpus | All 5 areas |

### Publication Strategy

| Paper | Focus | Target Venue | Timeline |
|-------|-------|--------------|----------|
| **Paper 1** | Token vs. Sequence + Task-Adaptive | ACL/EMNLP | Month 4 |
| **Paper 2** | Prefix × Temperature Interaction | NeurIPS/ICML | Month 7 |
| **Paper 3** | Curriculum Scheduling | TACL/JMLR | Month 9 |
| **Paper 4** | Model Collapse Prevention | Nature Machine Intelligence | Month 13 |
| **Paper 5** | Comprehensive Survey & Guidelines | JAIR/AI Magazine | Month 15 |

### Open-Science Deliverables

1. **Code Repository:** Full implementation of EDT for generative replay
2. **Pre-generated Corpus:** 500,000+ synthetic samples across all conditions
3. **Benchmark Suite:** Standardized metrics for future research
4. **Interactive Dashboard:** Visualization of collapse trajectories
5. **Best Practices Guide:** Implementation recommendations for practitioners

---

*This research planning document was generated to support systematic investigation of dynamic temperature sampling for generative replay. All proposed methodologies should be validated through pilot studies before full-scale implementation.*