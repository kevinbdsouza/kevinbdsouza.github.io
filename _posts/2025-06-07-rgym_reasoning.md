---
title: 'Stress-Testing LLMs With Reasoning Gym: Building & Training a Multi-step Reasoning Task'
date: 2025-06-07
permalink: /posts/2025/06/rgym-reasoning
tags:
  - Reasoning Gym 
  - Reinforcement Learning  
  - Tool Calling
---

I’ve been exploring how far reinforcement-learning paradigms can push large language models when the reward is verifiable reasoning correctness. That led me to (i) extending Reasoning Gym with a procedurally-generated, multi-hop puzzle set that forces deduction ↔ induction ↔ abduction ↔ transduction hand-offs, (ii) wiring it into the TRL training loop, and (iii) seeing what the first accuracy curves look like. Below is the why, the how, and the initial results.

<style>
/* affects only this file */
code             { font-size: 12px;}
pre, pre code    { font-size: 12px;}
</style>

## Why another task?
Reasoning Gym already ships with over one hundred single-skill datasets. But real-world reasoning is messy; it rarely stays in a single mode for more than a sentence or two. I wanted a benchmark that could push models to navigate more complex, multi-stage problems.

The design goals were:

  * **Mixed modes:** Force the policy to transfer intermediate state between fundamentally different reasoning operations. Each puzzle is a chain of 5-10 steps, randomly drawn from deduction, induction, abduction, and transduction.
  * **State persistence:** Later steps must depend on what the model actually inferred earlier. A shared state dictionary is threaded through the puzzle generator, with each operation mutating it.
  * **Reward = truth:** The model gets a reward of 1.0 if the final answer is provably correct, and 0.0 otherwise. This is handled by a simple, exact-match scoring function.
  * **Curriculum-ready:** Expose minimum and maximum chain length as simple configuration knobs to allow for annealing complexity during training.

## Implementing the Multi-step Reasoning Task
The dataset generator is a concise \~150 lines of Python. The core logic involves:

1.  A `@dataclass` to hold the configuration (`min_steps`, `max_steps`, etc.).
2.  Four helper methods (`_deduction`, `_induction`, `_abduction`, `_transduction`), each of which takes the current state, performs a transformation, and returns a new natural-language prompt.
3.  A public `__getitem__` method that stitches together a random sequence of these steps and appends a final "What is the answer?" query.

Because every answer can be deterministically derived from the state, verification is a straightforward and constant-time operation. Check out the implementation [here](https://github.com/kevinbdsouza/reasoning-gym/blob/main/reasoning_gym/logic/multi_step_reasoning.py). 

## Testing with the TRL GRPOTrainer
I modified the existing minimal script for trl in the Reasoning Gym repo, the one found [here](https://github.com/kevinbdsouza/reasoning-gym/blob/main/examples/trl/main_grpo_reward.py). The very first run with a DeepSeek-R1-Distill-Qwen-1.5B parameter model yielded promising results. The W\&B plot for the mean accuracy reward shows a clear and steady improvement throughout training (Fig. 1).

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/reward_rgym.png?raw=true" width="600"/>
</p>
<p align="center">
<em> <font size="2"> Fig. 1: Train mean accuarcy reward.</font> </em>
</p>

## Multi-Hop Tool Calling Might Help
Allowing the model to call a Python interpreter or a calculator for heavy lifting seems like a promising direction for this type of reasoning task. While I don't have results for it yet, I've set up the `ToolEnv` from the `verifiers` library to do this, see [here](https://github.com/kevinbdsouza/verifiers/blob/main/verifiers/examples/multi_step_reasoning_tools.py). 

## Closing Thoughts

`multi_step_reasoning` turns Reasoning Gym into a mini-IMDB for causal chains: a procedurally generated catalogue of plots where the protagonist (the model) has to keep track of its own deductions if it ever hopes to resolve the ending. The early signs are promising—especially when we think about handing the model tools. 