<style>
/* affects only this file */
code             { font-size: 0.95rem;}
pre, pre code    { font-size: 1.05rem;}
</style>

---
title: 'Teaching a 1.5-Billion-Parameter LLM to Classify Land-Use Decisions with RLVR and Spatial Heuristics'
date: 2025-04-12
permalink: /posts/2025/03/rlvr-classification
tags:
  - artificial intelligence 
  - reinforcement learning 
  - large language models
  - heuristics 
---

When I asked whether a compact 1.5-B parameter model could double as a local land-use classifier, I was really probing two things at once:
1. Expressive power – do today’s distilled language models understand enough geography and have enough spatial awareness to be decision makers? 
2. RLVR – can reinforcement learning from verifiable rewards (RLVR) scale beyond toy domains?

## From global optimum to conversational apprentice
I start with a mixed-integer optimisation model, the Global Land Manager (GLM), that finds the yield-connectivity optimum for an entire Canadian landscape. This solution acts as an oracle:

$$Quadrant (x, y) → {habitat, crop}$$

Because every label is verifiable, we can define a crisp reward:

| component          | symbol     | description                                        | weight |
| ------------------ | ---------- | -------------------------------------------------- | ------ |
| Land-use match     | `R_LU`     | 1 if the heuristic’s `<answer>` tag matches oracle | **3**  |
| Format bonus       | `R_Format` | well-formed `<think>` / `<answer>` blocks          | **2**  |
| Repetition penalty | `R_Repeat` | − 1 × (unique n-gram count)                        | **1**  |

I use **DeepSeek-R1-Distill-Qwen-1.5B** with **Group Relative Policy Optimisation (GRPO)**, which clips the KL-divergence to a reference model. Whenever I zero-out the KL anchor (`β = 0`), reward spikes and then falls as the policy latches onto easy-to-game formats and degenerates into repetition (Fig. 1).

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/no_kl.png?raw=true" width="600"/>
</p>
<p align="center">
<em> <font size="2"> Fig. 1: Train rewards without KL term.</font> </em>
</p>


## Implementation details

I borrow the <a href="https://github.com/huggingface/open-r1"><u>open-r1</u></a> implementation from hugingface. A Slurm file launches the grpo training run using accelerate, modified from <a href="https://github.com/huggingface/open-r1/blob/main/slurm/train.slurm"><u>here</u></a>. Here’s the trimmed-down version:

```bash
#!/bin/bash -l
#SBATCH --job-name=farm_grpo
#SBATCH --time=4-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64000           
#SBATCH --gpus-per-node=a100:4

module load cuda
source openr1/bin/activate

echo "START TIME: $(date)"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

...

export CMD=" \
    src/open_r1/grpo.py --config $CONFIG_FILE $OPTIONAL_ARGS
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=0 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "

...

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1
```

Inside **`grpo.py`**, I specify custom rewards:

```python
  REWARD_FUNCS_REGISTRY = {
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "lu_reward": lu_reward,
        "format_reward": format_reward
    }
  reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
```

and the heavy lifting (advantage estimation, adaptive KL, LR decay) is done by `GRPOTrainer`.

```python
torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
model_kwargs = dict(
    revision=model_args.model_revision,
    trust_remote_code=model_args.trust_remote_code,
    attn_implementation=model_args.attn_implementation,
    torch_dtype=torch_dtype,
    use_cache=False if training_args.gradient_checkpointing else True,
)
training_args.model_init_kwargs = model_kwargs

#############################
# Initialize the GRPO trainer
#############################
trainer = GRPOTrainer(
    model=model_args.model_name_or_path,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=get_peft_config(model_args),
    callbacks=get_callbacks(training_args, model_args),
    processing_class=tokenizer,
    )
```

I use the following parameters in the config yaml:

```yaml
torch_dtype: bfloat16
attn_implementation: flash_attention_2
bf16: true
use_vllm: true
vllm_device: auto
vllm_gpu_memory_utilization: 0.6
do_train: true
do_eval: true
eval_strategy: "steps"
eval_steps: 1000
gradient_accumulation_steps: 4
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
learning_rate: 1.0e-06
log_completions: true
log_level: info
logging_first_step: true
logging_steps: 1
logging_strategy: steps
lr_scheduler_type: cosine_with_min_lr
lr_scheduler_kwargs:
  min_lr_rate: 0.1
max_prompt_length: 5000
max_completion_length: 4096
max_steps: -1
num_generations: 4
num_train_epochs: 4
per_device_eval_batch_size: 4
per_device_train_batch_size: 4
push_to_hub: false
report_to:
- wandb
reward_funcs:
- lu_reward
- format_reward
- repetition_penalty
reward_weights:
- 3.0
- 2.0
- 1.0
save_strategy: "steps"
save_total_limit: 1
seed: 42
temperature: 0.7
warmup_ratio: 0.1
```

and the following prompt template:
```
<|begin▁of▁sentence|>You are a helpful assistant …
System pre-amble
  └─ declares the assistant an expert in spatial optimisation / land-use planning  
     and asks for chain-of-thought inside <think> … </think> tags.

Formatting rules
  • Think step-by-step **inside** <think> … </think>  
  • Put **only** the chosen land-use class inside <answer> … </answer>  
  • Allowed answers: wheat | oat | habitat | corn | soy  
  • “Strictly adhere to this format. Don’t output anything else.”

User instruction block
  └─ Re-states the task in second person and explains:
     – position to be decided (`central`)  
     – semantic meaning of the variables (`*_yield`, `ecological_connectivity`)  
     – neighbourhood graph (N, S, E, W, NE, … + second-order d1-d2 codes)  
     – high-level optimisation goal  
       ▸ maximise *global* connectivity while keeping landscape-level crop supply

Input data
  └─ JSON-like list of dictionaries, one per polygon
     { position, oat_yield, wheat_yield, corn_yield, soy_yield, ecological_connectivity }
<|Assistant|>
```

## Rewards and completion lengths

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/reward_epochs.png?raw=true" width="600"/>
</p>
<p align="center">
<em> <font size="2"> Fig. 2: Train reward (moving-average) across steps.</font> </em>
</p>

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/comp_len.png?raw=true" width="600"/>
</p>
<p align="center">
<em> <font size="2"> Fig. 3: Completion length across steps.</font> </em>
</p>

The run begins in the gutter (≈ 0.4), tanks further as the policy struggles with formatting, then climbs steadily once the `<think>` / `<answer>` template locks in. It tops out around **3.2** by 14 k steps (Fig. 2: Epoch 1). Starting from the previous checkpoint, the initial reward is already ≈ 3.4. A short “digestion dip” follows (the policy adapts to the new learning-rate schedule) before reward recovers and plateaus just under **3.6** (Fig. 2: Epoch 2). The curve is almost flat; reward hovers around **4.1 ± 0.1** with only stochastic noise. At this point all three reward components are saturated (Fig. 2: Epoch 4). Early in Epoch 1 completions average **≈ 1 000 tokens**, peaking at a verbose **1 300–1 500** when the policy “hallucinates” long-winded justifications (Fig. 3). As rewards improve, the average shrinks to **≈ 500 tokens**. By Epoch 4 the model stabilises at **≈ 300–350 tokens** per sample—just enough for a concise `<think>` plus a single-word `<answer>` (Fig. 3). Shorter completions correlate with higher reward because (i) the repetition penalty bites less, and (ii) long chains of reasoning are rarely needed for a quadrant decision once the local-neighbour rule is known.

Mapping answers back to class-labels gives a **\~ 65 %** classification accuracy on a held out test set. Together, this suggests that training proceeds not by bloating text but by tightening it: as heuristics become crisper, the policy earns more reward with fewer tokens, yet still lacks enough context to resolve the trickiest edge cases. 

## Recurring strategies & heuristics in the completions 

| # | Strategy / Heuristic                                                                                                                                                                    | What it looks like in the think monologue                                                               | Frequency\* |
| - | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------- |
| 1 | **Yield-max first pass** – rank candidate crops by the *central* polygon’s t ha⁻¹ yield and keep only the top-1 or top-2.                                                               | “Wheat gives 0.74 t ha⁻¹, oat 0.56; corn/soy are zero ⇒ ignore them.”                                       | \~85 %      |
| 2 | **Connectivity tie-breaker** – when two crops have comparable yield, choose the one whose adoption keeps local/ecoregion connectivity ≥ current level.                                  | “Both wheat and oat similar here, but oat neighbours already give 0.72 connectivity corridor – choose oat.” | \~60 %      |
| 3 | **‘Habitat if zero-yield belt’ rule** – if the central polygon *and* ≥ 50 % of its neighbours have 0-yield for all crops, flip to *habitat* to consolidate a corridor.                  | “Central and three neighbours are already zero-yield → converting to habitat stitches the patch.”           | \~25 %      |
| 4 | **Neighbourhood averaging** – compute mean or max crop yield across 1-hop neighbours and penalise options that would introduce sharp yield discontinuities (crop-rotation realism).     | “Neighbour wheat mean = 1.74 > central 0.74 ⇒ wheat keeps agronomic coherence.”                             | \~20 %      |
| 5 | **Global baseline check** (mentioned, not computed) – brief reminder that landscape-wide quotas exist; used rhetorically to justify staying with a high-yield crop rather than habitat. | “Need to maintain wheat supply globally, so habitat not wise here.”                                         | \~15 %      |
| 6 | **Simple scoring function** – form a weighted score `S = α·yield + β·connectivity` with α≈1, β≈0.5 (weights rarely explicit) and pick argmax S.                                         | “Score\_wheat = 0.74 + 0.34; Score\_oat = 0.56 + 0.34 … wheat wins.”                                        | \~10 %      |
| 7 | **Corridor extension logic** – if an adjacent polygon is already habitat and opposite neighbour has high connectivity, decide in favour of habitat to ‘bridge the gap’.                 | “North is habitat, south connectivity = 0.72; turning central into habitat closes the corridor.”            | \~8 %       |

\*Percentages are approximate share of completions in which the motif appears (multiple motifs often co-occur).


## Typical reasoning flow inside `<think> … </think>`

1. **Data parsing** – restate or tabulate central vs. neighbour yields and connectivity.
2. **Prune impossible options** – drop crops with zero yield.
3. **Primary metric** – select the highest-yield crop for the central cell.
4. **Secondary check** – adjust for ecological connectivity (threshold ≈ 0.65–0.70).
5. **Edge-case rules** – apply habitat or rotation heuristics if special patterns detected.
6. **Explain choice briefly** – one-sentence justification.
7. **Close tag** – `</think>` followed by `<answer>{crop}</answer>`.

The model almost always honours the required tag structure, uses a concise single-word answer, and avoids leaking the chain-of-thought outside the `<think>` block.

## Discussion

How does this compare with decision trees for interpretable rules? Decision trees are simpler, cheaper, and structurally transparent, but they can neither converse with planners nor ingest messy qualitative context the way an LLM can. In practice the language model’s `<think>` block gives domain-specific planners a narrative explanation they can accept, critique, or refine—capabilities that a static rule list cannot match. Heuristics remain simple (one- or two-factor scoring, local neighbourhood reasoning); yet the variety of tie-breakers (habitat corridor logic, global baseline reminder) introduces diversity that RLVR can exploit.

Even at 1.5 B parameters the model costs real GPU time, but the trade-off is a conversational agent that captures 65 % of the oracle’s decisions and—crucially—speaks its reasoning out loud. That makes the answer to our opening question a guarded *yes*: with RLVR providing a verifiable signal, a smallish LLM can indeed learn to classify spatial decisions while explaining itself in plain language. 


