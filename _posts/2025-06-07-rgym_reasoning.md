---
title: 'Stress-Testing LLMs With Reasoning Gym: Building & Training a Multi-step Reasoning Task'
date: 2025-06-07
permalink: /posts/2025/06/rgym-reasoning
tags:
  - Reasoning Gym 
  - Reinforcement Learning  
  - Tool Calling
  - Large Language Models 
---

I’ve been exploring how far reinforcement-learning paradigms can push large language models when the reward is verifiable reasoning correctness. That led me to (i) extending Reasoning Gym with a procedurally-generated, multi-hop puzzle set that forces deduction ↔ induction ↔ abduction ↔ transduction hand-offs, (ii) wiring it into the TRL training loop, and (iii) seeing what the first accuracy curves look like. Below is the why, the how, and the initial results.

<style>
/* affects only this file */
code             { font-size: 12px;}
pre, pre code    { font-size: 12px;}
</style>

## Why Another Task?
[Reasoning Gym](https://github.com/open-thought/reasoning-gym) already ships with over one hundred tasks, but I noticed that it doesn't have a task that chains different modes of reasoning together. Real-world reasoning is messy; it rarely stays in a single reasoning mode for more than a sentence or two. Therefore, I wanted a benchmark that could push models to navigate more complex, multi-stage problems.

The design goals were:

  * **Mixed modes:** Force the policy to transfer intermediate state between fundamentally different reasoning operations. Each puzzle is a chain of 5-10 steps, randomly drawn from deduction, induction, abduction, and transduction.
  * **State persistence:** Later steps must depend on what the model actually inferred earlier. A shared state dictionary is threaded through the puzzle generator, with each operation mutating it.
  * **Reward = truth:** The model gets a reward of 1.0 if the final answer is provably correct, and 0.0 otherwise. This is handled by a simple, exact-match scoring function.
  * **Curriculum-ready:** Expose minimum and maximum chain length as simple configuration knobs to allow for annealing complexity during training.

## Why These Four Modes, and Why Chain Them?

| Reasoning Mode   | Cognitive Role it Plays                                        | Typical Failure Pattern in LLMs                                      |
| ---------------- | -------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Deduction**    | Certainty: deriving a consequence that *must* hold.           | Skips a premise or swaps quantifiers.                                |
| **Induction**    | Generalisation: inferring a rule from a few concrete cases.   | Over- or under-generalises; anchoring on surface cues.               |
| **Abduction**    | Explanation: choosing the best hypothesis for an observation. | “Explanation hacking”: proposes a plausible but *non-minimal* cause. |
| **Transduction** | Analogy/Translation: mapping patterns across domains.         | Breaks when the structural mapping is deep rather than lexical.      |

Each mode feels orthogonal because the *direction* of inference flips:

* **Deduction:** rules → instance
* **Induction:** instances → rule
* **Abduction:** effect → cause
* **Transduction:** structure ↔ structure

Real tasks ping-pong between these directions. Example: to debug scientific code you (a) **deduce** what output *should* be, (b) **abduce** which line could cause the mismatch, (c) **induce** a new rule from several failures, and (d) **transduce** the fix to a different module. A benchmark that stalls in a single mode never pressures a model to learn those hand-offs.

**Chaining them forces three things at once**

1. **Working memory under adversarial conditions.**
   The state emitted by an inductive step may be probabilistic, yet the next deductive step demands a crisp proposition. Forgetting or mis-typing that intermediate state poisons the rest of the chain and yields a zero reward.
2. **Meta-reasoning.**
   The policy has to notice when it is leaving the comfort zone of one inference style and switch tool-sets. That is exactly the emergent capability we want from LLM agents writing code, orchestrating tools, or giving medical advice.
3. **Fine-grained diagnostics.**
   Because every sub-step is labelled, we get a confusion matrix over modes rather than a single-bit accuracy flag. If a run collapses whenever abduction is step #3, we have a clear research question: *why does hypothesis selection fail only when the context window is half-full?*

## Implementing the Multi-step Reasoning Task
The dataset generator is a concise \~150 lines of Python. The core logic involves:

1.  A `@dataclass` to hold the configuration (`min_steps`, `max_steps`, etc.).
2.  Four helper methods (`_deduction`, `_induction`, `_abduction`, `_transduction`), each of which takes the current state, performs a transformation, and returns a new natural-language prompt.
3.  A public `__getitem__` method that stitches together a random sequence of these steps and appends a final "What is the answer?" query.

Because every answer can be deterministically derived from the state, verification is a straightforward and constant-time operation. Check out the implementation [here](https://github.com/kevinbdsouza/reasoning-gym/blob/main/reasoning_gym/logic/multi_step_reasoning.py). See a small excerpt below. 

```python
class MultiStepReasoningDataset(ProceduralDataset):
    """Dataset generating multi-step puzzles mixing deduction, induction, abduction and transduction."""

    WORD_BANK = [
        "lion",
        "tiger",
        "bear",
        "wolf",
        "eagle",
        "shark",
        "horse",
        "whale",
        "otter",
        "camel",
    ]

    NAME_BANK = [
        "Alice",
        "Bob",
        "Carol",
        "Dave",
        "Eve",
        "Frank",
        "Grace",
        "Heidi",
        "Ivan",
        "Judy",
    ]

    def __init__(self, config: MultiStepReasoningConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = Random(self.seed + idx if self.seed is not None else None)
        return self._generate_item(rng, idx)

    def _deduction(self, step_no: int, rng: Random, state: dict) -> str:
        """Make a deterministic reasoning step that requires deduction."""
        choice = rng.random()
        if choice < 0.33 or state.get("word") is None:
            mult1 = rng.randint(10, 20)
            add = rng.randint(5, 15)
            mult2 = rng.randint(2, 5)
            sub = rng.randint(2, 10)
            res = (state["num"] * mult1 + add - sub) * mult2
            line = (
                f"Step {step_no}: Multiply {state['num']} by {mult1}, add {add}, subtract {sub}, "
                f"then multiply the result by {mult2}. What number results?"
            )
            state["num"] = res
        elif choice < 0.66:
            short_thres = rng.randint(3, 4)
            long_thres = rng.randint(7, 9)
            vowel_count = sum(1 for c in state["word"] if c in "aeiou")
            if len(state["word"]) <= short_thres or vowel_count < 2:
                classification = "small"
            elif len(state["word"]) >= long_thres and vowel_count >= 3:
                classification = "large"
            else:
                classification = "medium"
            line = (
                f"Step {step_no}: Words with \u2264{short_thres} letters or fewer than 2 vowels are 'small'; "
                f"those with \u2265{long_thres} letters and at least 3 vowels are 'large'; otherwise 'medium'. "
                f"Is '{state['word']}' small, medium, or large?"
            )
            state["word"] = classification
        else:
            a, b, c, d = rng.sample(self.NAME_BANK, 4)
            line = (
                f"Step {step_no}: {a} and {b} are siblings. {b} is {c}'s parent and {c} and {d} are siblings. "
                f"Who is {d}'s aunt or uncle?"
            )
            state["person"] = a
        return line

    def _induction(self, step_no: int, rng: Random, state: dict) -> str:
        """Make an inductive reasoning step with slightly harder patterns."""
        choice = rng.random()
        if choice < 0.33:
            mult = rng.randint(2, 4)
            add = rng.randint(3, 7)
            n = rng.randint(3, 5)
            value = state["num"]
            for _ in range(n):
                value = value * mult + add
            line = (
                f"Step {step_no}: Starting at {state['num']}, repeatedly multiply by {mult} and add {add} "
                f"for {n} iterations. What number results?"
            )
            state["num"] = value
        elif choice < 0.66:
            new_word = state["word"][::2][::-1] + state["word"][-1]
            line = (
                f"Step {step_no}: Take every second letter of '{state['word']}', reverse those letters, "
                f"and append the last letter of the original word. What word results?"
            )
            state["word"] = new_word
        else:
            start = state.get("person", rng.choice(self.NAME_BANK))
            forward = rng.randint(1, 3)
            backward = rng.randint(1, 3)
            n = rng.randint(2, 4)
            idx = self.NAME_BANK.index(start)
            for _ in range(n):
                idx = (idx + forward) % len(self.NAME_BANK)
                idx = (idx - backward) % len(self.NAME_BANK)
            target = self.NAME_BANK[idx]
            line = (
                f"Step {step_no}: Starting from {start}, move forward {forward} names then backward {backward} names, "
                f"repeating this {n} times in {self.NAME_BANK}. Which name do you reach?"
            )
            state["person"] = target
        return line
``` 

## Testing with the TRL GRPOTrainer
I modified the existing minimal script for trl in the Reasoning Gym repo, the one found [here](https://github.com/kevinbdsouza/reasoning-gym/blob/main/examples/trl/main_grpo_reward.py). The very first run with a `DeepSeek-R1-Distill-Qwen-1.5B` model yielded promising results. The plot for the mean accuracy reward shows a clear and steady improvement throughout training (Fig. 1).

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/reward_rgym.png?raw=true" width="600"/>
</p>
<p align="center">
<em> <font size="2"> Fig. 1: Train mean accuarcy reward.</font> </em>
</p>

## Multi-Hop Tool Calling Might Help
Allowing the model to call a Python interpreter and calculator for heavy lifting seems like a promising direction for this type of reasoning task. While I don't have results for this yet, I've set up the `ToolEnv` from the `verifiers` library to do this, see [here](https://github.com/kevinbdsouza/verifiers/blob/main/verifiers/examples/multi_step_reasoning_tools.py) and an excerpt below. 

```python
rg_env = ReasoningGymEnv(
    gym="multi_step_reasoning",
    num_samples=2000,
    num_eval_samples=200,
    max_concurrent=128,
)

vf_env = ToolEnv(
    dataset=rg_env.dataset,
    eval_dataset=rg_env.eval_dataset,
    system_prompt=DEFAULT_TOOL_PROMPT_TEMPLATE,
    few_shot=[],
    tools=[python, calculator],
    max_turns=5,
)
``` 

## Closing Thoughts
`multi_step_reasoning` tries to benchmark causal chains of reasoning modes and transitions between them. Early results show that even a 1.5 B parameter model can be trained to perform this dance in its CoT. The next milestones are:

1. **Tool fluency.** Let the agent decide whether to think, to call Python, or to delegate to a symbolic prover.
2. **Long-horizon chains.** Bump the maximum length to 25–30 steps and watch for compositionality.
3. **Failure-mode atlases.** Analyse per-mode heat maps so we can compare how different reasoning modes fail relative to one another, and how often they fail.






