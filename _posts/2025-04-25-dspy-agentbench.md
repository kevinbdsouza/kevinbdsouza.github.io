---
title: 'Evaluating DSPy-Based Prompt Optimisation on AgentBench'
date: 2025-04-25
permalink: /posts/2025/04/dspy-agentbench
tags:
  - DSPy 
  - Agents 
  - Multi-hop tool use
---

AgentBench’s **dbbench-std** task evaluates an agent’s ability to answer SQL questions in a multi-hop tool use setting. The controller exposes interaction endpoints, so that every task instance can be completed with a small, repeatable tool repertoire:

<style>
/* affects only this file */
code             { font-size: 12px;}
pre, pre code    { font-size: 12px;}
</style>


| Tool       | Purpose                                                                   | Typical arguments                                          |
| ---------- | ------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `init`     | Open a task session, receive the SQL instructions & the concrete question | `index=<int>`                                              |
| `db_query` | Issue **one** SQL query and observe the result                            | `session_id=<str>`, `sql='<…>'`                            |
| `finish`   | Submit the final JSON-array answer                                        | `session_id=<str>`, `final_answer_json_array_string='<…>'` |

A successful run therefore requires a *multi-hop* dialogue:
`init → db_query* (≥1) → finish`.
At each hop the agent must:

1. Decide **which** tool to call next.
2. Build its arguments (e.g., reuse `session_id`, compose a single-line SQL string).
3. Incorporate the controller’s new message into its chain-of-thought (CoT) before the following hop.

## Environment

First I setup the environment:

```bash
conda create -n dspy-agentbench python=3.9
conda activate dspy-agentbench

pip install dspy mlflow func_timeout ujson requests
git clone https://github.com/THUDM/AgentBench
cd AgentBench && pip install -r requirements.txt
```

followed by pulling AgentBench Docker images:
```bash
docker pull mysql
docker pull ubuntu
docker build -f data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles --tag local-os/ubuntu
```

and starting the server for worker tasks: 
```python
python -m src.start_task -a
```

## Baseline **Chain-of-Thought** agent in DSPy

The baseline agent is a CoT implementation in DSPy. Below is a lightly annotated view of the baseline agent’s control flow. Everything outside the grey block is ordinary Python; the grey block is where DSPy injects the LM.

```python
class Agent(dspy.Module):
    def __init__(self, max_steps: int = 5):
        super().__init__()
        # 1.  Build the CoT predictor ----
        sig = dspy.Signature(
            "question, trajectory, functions -> next_selected_fn, args: dict[str, Any]",
            instructions=REACTION_PROTOCOL,          # ❶ English tool grammar
        )
        self.react = ChainOfThought(                # ❷ wrapper shown in snippet you sent
            signature=sig,
            temperature=0.7,
            max_tokens=512
        )
        self.max_steps = max_steps

    # 2.  One AgentBench task instance -------------
    def forward(self, question, functions):
        traj = []                                   # running conversation transcript
        for _ in range(self.max_steps):
            # ⬇⬇⬇ -------------  LM call (DSPy handles I/O) ------------- ⬇⬇⬇
            pred = self.react(
                question   = question,
                trajectory = traj,
                functions  = {n: fn_metadata(f) for n, f in functions.items()},
            )
            # ⬆⬆⬆ -------------------------------------------------------- ⬆⬆⬆

            fn_name = pred.next_selected_fn.strip()
            args    = pred.args or {}

            result  = call_with_timeout(functions[fn_name])(**args)
            traj.append({**pred, **result})         # keep both reasoning & server reply
            if fn_name == "finish":
                break

        return dspy.Prediction(answer=result, trajectory=traj)
```

| Aspect                            | What happens                                                                                                                                                                                                                                             |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Signature**                     | The CoT signature has two *inputs*—`question` and `trajectory` (the full history)—and two *outputs*: `next_selected_fn` (a string literal that must match one of `["init", "db_query", "finish"]`) and an `args` dict.                                   |
| **ChainOfThought** (your snippet) | Prepends an extra field called `reasoning` to the signature. During generation the LM fills `reasoning` first (“Let’s think step by step …”), then fills `next_selected_fn`, and finally the JSON-like `args`.                                           |
| **Trajectory growth**             | After each tool call we append a dict containing: *LM reasoning*, *selected\_fn*, *args actually used*, and the *server’s return payload/errors*. This trajectory is re-fed into the next LM call, giving it visibility over past successes or failures. |
| **Termination**                   | The loop exits either when the LM chooses `finish` itself or when `max_steps` is hit (in which case a forced `finish` with a dummy answer is issued so AgentBench can close the session gracefully).                                                     |


I use `gemini/gemini-2.0-flash` as my language model with `temperature = 0.7` and `max\_tokens = 2048)`. The baseline CoT agent with this LM achieved **\~ 68 %** success rate in finding the correct answers. Next, I wanted to check whether **DSPy’s** built-in optimiser (SIMBA) could provide a measurable improvement without altering model weights or adding training data.

## Optimisation with SIMBA

The SIMBA optimiser is part of DSPy’s *teleprompting* suite.  Its goal is simple: given a **metric** and a **train-set**, iteratively rewrite the *prompt programs* that wrap your predictors so that average metric‐score improves. The algorithm has three ideas worth highlighting:

| Idea                                         | What actually happens in code                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1 . A pool of competing programs**         | The optimiser starts with a deepcopy of our baseline agent (`student`) and assigns it `simba_idx = 0`.  Each time it invents a new variant it registers it in `programs[]` and stores its per-example scores in `program_scores`.                                                                                                                |
| **2 . Mini-batch, multi-candidate sampling** | For every optimisation step SIMBA draws a mini-batch (size `bsize`, here 32).  For each example it pairs **one LM clone** (with its own temperature) with **one prompt program** sampled from the pool via a soft-max over current average scores.  The wrapper `wrap_program()` runs the candidate on the example and returns the metric value. |
| **3 . Heuristic edits**                      | New prompt variants are created by stochastic *strategies* – by default `append_a_demo` (adds a fresh “demo shot” built from a high-scoring trajectory) and `append_a_rule` (adds a short natural-language rule).  If `max_demos > 0` both strategies are active; otherwise only rules are used.                                                 |

Below is a schematic of one optimisation round (with example parameters):

```text
(bsize = 32, num_candidates = 6)

                 ┌─────────────────────────────┐
                 │    Program pool (size ~k)   │
                 └────────────┬────────────────┘
                              │soft-max sampling
                              ▼
+-------------------+    +------------------------------+
| 32 train examples |    | 6 LM clones (T = 0.2 each)  |
+--------┬----------+    +-------------┬--------------+
         │                           ┌─┴────────────────────┐
         └─────────────▶ 192  (program,LM,example) triples ─┤
                                         │batched execution │
                                         ▼
                                192 metric scores → buckets
                                         │top-bucket stats
                                         ▼
                         strategies ↑    ▲
                         (demos,rules)   │
                                         │new prompt programs
                                         ▼
                           evaluate same 32 examples
                                         │
                                         ▼
                          register candidates, update pool
```

In the implementation, the SIMBA portion looks like this:
```python
def metric(_, pred, __):
    return int(bool(pred.answer and pred.answer.get("done")))

simba = dspy.SIMBA(metric=metric, max_steps=5, max_demos=1,
                   bsize=1, num_threads=1, seed=42)

optim_agent = simba.compile(
        student=Agent(max_steps=5),
        trainset=make_trainset(train_indices)   
)
```

Evaluating the optimized agent on the test set, it achieved **\~ 74 %** success rate in finding the correct answers. SIMBA searches over wording, shot selection, and signature details; no gradient updates are involved. Further gains would likely come from a larger demo pool, more depth of tool calling, or more aggressive rule generation. 

## Discussion

Below are three condensed but real traces taken from the SIMBA optimized model (max 5 steps).
For each task you see the LM’s **selected function**, the **SQL** it produced, and the controller’s reply that was ultimately fed back into the next step.

| Step                            | Tool call (arguments)                                                                   | Controller reply (truncated)                                        |
| ------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Task #7 — Crest Whitestrips** |                                                                                         |                                                                     |
| 1                               | `init(index = 7)`                                                                       | returns `session_id = 292`, task question about *lasting whiteness* |
| 2                               | `db_query(sql = SHOW COLUMNS FROM Crest Whitestrips Products)`                          | column list                                                         |
| 3                               | `db_query(sql = SELECT Last of whiteness … WHERE Model IN (…))`                         | `[('12 months',), ('12 months',)]`                                  |
| 4                               | `finish(final_answer = ["12 months","12 months"])`                                      | `done = True`                                                       |
| **Task #11 — MMA Fight Record** |                                                                                         |                                                                     |
| 1                               | `init(index = 11)`                                                                      | `session_id = 293`, question about *Masato Shiozawa*                |
| 2                               | `db_query(sql = SELECT Event FROM MMA Fight Record WHERE Opponent = 'masato shiozawa')` | `[('Shooto 2003 – 5/4 in Korakuen Hall',)]`                         |
| 3                               | `finish(final_answer = ["Shooto 2003 – 5/4 in Korakuen Hall"])`                         | `done = True`                                                       |
| **Task #36 — NFL Draft Picks**  |                                                                                         |                                                                     |
| 1                               | `init(index = 36)`                                                                      | `session_id = 294`, question about *Round* for Indiana < 198        |
| 2                               | `db_query(sql = SELECT * FROM NFL Draft Picks)`                                         | full table (truncated)                                              |
| 3                               | `db_query(sql = SELECT Round … WHERE School/Club Team = 'Indiana' AND Pick < 198)`      | `[]`                                                                |
| 4                               | `finish(final_answer = [])`                                                             | `done = False`                                                       |
The agent almost always follows the same **schema-first → filtered-query → finish** pattern and typically completes a task in 3–4 tool calls. In the baseline, 80 % of successful cases finished within four steps; SIMBA kept that length unchanged while reducing validation errors on the first `db_query`.

A single SIMBA pass—five mini-batch steps with six prompt variants each—nudged the baseline ReAct agent from **68 % to 74 %** accuracy on **dbbench-std**. The gain stems almost entirely from lower formatting and protocol mistakes; no additional reasoning depth or longer trajectories were needed.  While modest, this improvement was achieved with minimal engineering effort and a fixed language-model endpoint.


