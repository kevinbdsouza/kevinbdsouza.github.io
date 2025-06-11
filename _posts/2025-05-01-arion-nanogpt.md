---
title: 'Evaluating a Self-Tuning Version of Muon on the NanoGPT Speedrun'
date: 2025-05-01
permalink: /posts/2025/04/arion-nanogpt
tags:
  - Muon 
  - Optimizer 
  - AI
  - Deep Neural Networks 
---

For the better part of a decade, Adam has been the default optimizer for training deep learning models. But the ground is shifting. As we scale to massive models, a new family of geometry-aware optimizers, most notably **Muon** [1, 2], has emerged as a promising contender. The results from the `modded-nanogpt` [3] speedrun showed that by respecting the unique geometry of neural network layers, we could achieve faster and more efficient training. This is backed by simultaneous and follow up works like Scion [4], Modular Duality [5], Gluon [6], steepest descent under a particular norm and manifold [7, 8], and spectral condition for feature learning [9]. 

<style>
/* affects only this file */
code             { font-size: 12px;}
pre, pre code    { font-size: 12px;}
</style>

I was curious of this shift and dove into the latest optimization theory. While Muon's geometric update direction provides gains, it still relies on a globally-tuned learning rate. I wanted to check if by borrowing some concepts from latest research, we can create an optimizer that could determine its own step size dynamically, adapting to the unique landscape of each layer. I call this **Arion**, Adaptive-Radius with dualizatION. Arion is designed to work in tandem with the spectral condition for feature learning [9].

## The Spectral Condition for Feature Learning

Feature learning is the process where a network's hidden representations evolve meaningfully during training. If they change too little, the network effectively "freezes" and can't learn new patterns [9]. A key insight by Yang et al. [9] is that for feature learning to occur, the feature vectors `h_l` at each layer, and their updates `Δh_l`, must maintain a consistent "energy" relative to their dimension. Formally, their L2 norm should scale with the square root of the layer's width: $\|h_l\|_2 = \Theta(\sqrt{n_l})$.

To satisfy this, they derive the **Spectral Scaling Condition**. This principle states that the spectral norm of a layer's weight updates, `||ΔW||_*`, must scale like the square root of its fan-out to fan-in ratio:

$$\|\Delta W_\ell\|_* = \Theta\left(\sqrt{\frac{n_l}{n_{\ell-1}}}\right) = \Theta\left(\sqrt{\frac{\text{fan-out}}{\text{fan-in}}}\right)$$

If your optimizer's updates satisfy this condition, your network will learn features effectively.

## Using Muon with an Adaptive Radius  

I wanted to investigate whether complementary theoretical ideas can dualize the gradient [10], satisfy the spectral condition, and set a local landscape derived adaptive radius. 

1.  **Geometric Direction (from Muon):** Compute the momentum buffer `m` and find its nearest orthogonal matrix `s` using the Newton-Schulz iteration. This provides the correct geometric direction for an update in the spectral norm [7, 8].

2. **Spectral Condition:** Make the layer-specific learning rate `η_l` proportional to `fan-out / fan-in` according to [9] and a result of dualizing the gradient under the spectral norm [7, 8, 10].  
3.  **Adaptive Radius (from Gluon):** Use the adaptive radius derived from the **$(L^0, L^1)$-smoothness** model from Gluon [6]. This model gives us a principled way to compute a step size `t` that guarantees descent and adapts to local curvature.

## A Self-Tuning Step Size from First Principles

The core of Arion is its ability to find the right step size automatically, for each layer, at every single iteration. It modifies the globally-tuned `lr` to a dynamic radius `t` that emerges from a theoretical understanding of the loss landscapes of deep networks.

### A Model for the Landscape: $(L^0, L^1)$-Smoothness

The classic analysis of optimizers assumes a single, fixed smoothness constant `L` for the entire loss landscape. However, the recent **Gluon** paper by Riabinin et al. [6] shows this is an oversimplification for deep learning. They propose a more realistic model called **layer-wise $(L^0, L^1)$-smoothness**, defined by the inequality:

$$||\nabla_i f(X) - \nabla_i f(Y)||_{(i)*} \le (L_i^0 + L_i^1 ||\nabla_i f(X)||_{(i)*}) ||X_i - Y_i||_{(i)}$$

This means the landscape's curvature (how fast the gradient can change) is not constant. Instead, it's adaptive and proportional to the size of the gradient itself. Steep regions (large gradient norm) are "less smooth," while flatter regions (small gradient norm) are "smoother." Empirical evidence confirms this model holds for transformers, and often with $L_i^0 \approx 0$, which simplifies the local smoothness to be directly proportional to the gradient's dual norm:

`Local Smoothness` $\approx L^1 \cdot \|\nabla f\|_*$

### Deriving the Optimal Step Size

This improved model of the landscape gives us a tool to create a local quadratic model of the loss function that is more accurate than one from standard smoothness. According to the descent lemma derived from this model, the loss at the next step is bounded by:

$$f(X_{k+1}) \le f(X_k) + \langle \nabla f(X_k), \Delta X_k \rangle + \frac{\text{Local Smoothness}}{2} \|\Delta X_k\|^2$$

Our goal at each step is to choose our update `ΔX_k` to minimize the right-hand side of this equation, guaranteeing the largest possible descent. Our update consists of a direction `s` and a magnitude `t`, so `ΔX_k = -t ⋅ s`. Substituting this in gives:

$$f(X_{k+1}) \le f(X_k) - t \cdot \langle \nabla f(X_k), s \rangle + \frac{\text{Local Smoothness}}{2} (t^2 \cdot \|s\|^2)$$

Now, I use two key definitions from the theory of dual norms:
1.  The direction `s` is the `dualize` operator. The inner product of a vector and its dualized direction is, by definition, the dual norm itself: `⟨∇f, s⟩ = ||∇f||_*`.
2.  The direction `s` is a unit-norm operator in the primary norm, so `||s|| = 1`.

The inequality simplifies to a basic quadratic in `t`:

$$f(X_{k+1}) \le f(X_k) - t \cdot \|\nabla f(X_k)\|_* + \frac{\text{Local Smoothness}}{2} \cdot t^2$$

To get the biggest drop in loss, I find the value of `t` that minimizes this expression. This gives the optimal adaptive radius:

$$t_{optimal} = \frac{\|\nabla f(X_k)\|_*}{\text{Local Smoothness}} = \frac{\|\nabla f(X_k)\|_*}{(L^0 + L^1 \|\nabla f(X_k)\|_*)}$$

### From Theory to Practice: The Arion Formula

The final challenge is that the true $L^1$ constant is an unknown property of the landscape, and can't be used directly. I use practical, online estimators for the terms in the `t_optimal` formula.

$$t = \frac{\text{dual_norm}(m)}{\text{EMA}(\|\nabla f\|_F) + \epsilon}$$

* **Numerator: `dual_norm(m)`**: This is the **nuclear norm** of the momentum buffer `m`. I use the correct dual norm here because it represents the "reward" or potential for descent in our quadratic model. As I have an efficient way to calculate it (`(m * s).sum().abs()`), I use the theoretically correct quantity to ensure our step's "engine" is properly calibrated.

* **Denominator: `EMA(||∇f||_F)`**: This is the online estimator for the entire `Local Smoothness` term. I use an Exponential Moving Average of the **Frobenius norm** of the raw gradient (`g.norm()`) for two key reasons:
    1.  **It's a good proxy:** The Frobenius norm is a holistic measure of the gradient's magnitude, incorporating all singular values, much like the nuclear norm in the theoretical formula. It's a strong and stable indicator of local steepness.
    2.  **It's extremely fast:** `g.norm()` is a highly optimized, single-kernel operation. Since we are already in the realm of estimation for the denominator, I choose the proxy that is most computationally efficient.

This design gives us the best of both worlds: a theoretically pure numerator where it's cheap to be accurate, and a fast, robust, and well-motivated estimator for the denominator. This is how Arion learns its own learning rate.

## The Arion Implementation

This code is a drop-in replacement for the Muon optimizer in `modded-nanogpt`. It is designed for DDP settings and uses packed communication to reduce overhead.

```python
class Arion(torch.optim.Optimizer):
    """
    Adaptive-Radius with dualizatION (Arion).

    This optimizer is designed to merge Muon's geometric direction and spectral condition for feature learning with an adaptive step size derived from an online estimation of the (L0, L1)-smoothness constants.
    """
    def __init__(self, params, lr=1.0, beta=0.95, l1_ema_gamma=0.01, nesterov=True, ns_steps=5, rank=0, world_size=1, eps=1e-8):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, beta=beta, nesterov=nesterov, l1_ema_gamma=l1_ema_gamma, ns_steps=ns_steps, eps=eps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            # Add 1 to the size to pack the scalar adaptive radius `t`
            packed_size = size + 1
            b = torch.empty(world_size, packed_size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta, l1_ema_gamma, eps, nesterov, ns_steps = (
                group['lr'], group['beta'], group['l1_ema_gamma'], group['eps'], group['nesterov'], group['ns_steps']
            )
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():
                handle.wait()
                for i, p_world in enumerate(params_world):
                    packed_view = update_buffer_views[i]
                    s = packed_view[:-1].view_as(p_world) # The direction
                    t = packed_view[-1].item()             # The adaptive radius

                    # The final update incorporates the adaptive magnitude AND the architectural scaling
                    fan_out, fan_in = p_world.shape[0], p_world.shape[1] if p_world.ndim > 1 else 1
                    scaling_factor = (fan_out / fan_in)**0.5
                    p_world.add_(s, alpha=-lr * t * scaling_factor)

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state.setdefault(p, {})
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["L1_ema"] = torch.tensor(1.0, device=p.device)

                    m_buf: Tensor = state["momentum_buffer"]
                    l1_ema: Tensor = state["L1_ema"]

                    # Standard momentum update
                    m_buf.mul_(beta).add_(g, alpha=1 - beta)
                    m = m_buf
                    if nesterov: # Nesterov momentum
                        m = g.add(m_buf, alpha=beta)

                    # Estimate local smoothness via EMA of gradient's Frobenius norm
                    l1_ema.mul_(1 - l1_ema_gamma).add_(g.norm(), alpha=l1_ema_gamma)

                    # Direction `s` from orthogonalization (dualize operation)
                    s = zeropower_via_newtonschulz5(m, steps=ns_steps)

                    # Numerator: Nuclear norm (dual of spectral norm)
                    with torch.autocast(device_type='cuda', enabled=False):
                        dual_norm = torch.tensordot(m.float(), s.float())
                    
                    # Adaptive radius `t`
                    t = dual_norm / (l1_ema + eps)
                    
                    # Pack direction `s` and radius `t` into a single tensor for communication
                    packed_input = torch.empty(p.numel() + 1, dtype=torch.bfloat16, device=p.device)
                    packed_input[:-1] = s.flatten()
                    packed_input[-1] = t
                else:
                    p_example = params[base_i]
                    packed_input = torch.empty(p_example.numel() + 1, dtype=torch.bfloat16, device=p_example.device)

                if base_i > 0:
                    update_prev()
                
                handle = dist.all_gather_into_tensor(update_buffer, packed_input, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

```

## Discussion and Comparison with Muon 
After finalizing the `Arion` implementation, I pitted it against the optimized Muon baseline from the `modded-nanogpt` speed runs. The goal was to see if a theoretically-grounded, self-tuning step size could not only simplify tuning but also unlock faster convergence. I ran both optimizers on the standard NanoGPT benchmark for 5000 steps on a single node with two A100 40GB GPUs.

The results are humbling.

| Optimizer | Time to 5000 Steps | Validation Loss |
| :--- | :--- | :--- |
| **Muon (Baseline)** | **30.4 minutes** | **4.30** |
| **Arion** | 55.8 minutes | 4.35 |

Arion was nearly twice as slow and reached a slightly worse validation loss.

My initial hypothesis that a more precise, adaptive step size would lead to superior performance, was not borne out in this setting. The extra computation required to dynamically calculate the adaptive radius at every step proved to be a significant bottleneck, overwhelming any potential gains in convergence speed.

Some observations:

**1. The Spectral Condition is a Powerful Baseline.**
The primary lesson here is just how effective Muon's core recipe is. The work of Yang et al. [9] showed that enforcing the `sqrt(fan-out/fan-in)` scaling is critical for feature learning. Muon, with its orthogonal updates and a well-tuned learning rate, already implements a highly effective approximation of this condition. My results suggest that for a stable architecture like NanoGPT, getting this architectural scaling right accounts for the vast majority of the performance gains over traditional optimizers. The additional layer of fine-grained, step-by-step adaptivity offered by Arion provides diminishing returns that can't justify the computational overhead.

**2. There is a "FLOPs Budget" for Theory.**
Arion succeeded in its goal of removing layer-specific learning rate tuning. Its adaptive radius, derived from the $(L^0, L^1)$-smoothness model of Riabinin et al. [6] is theoretically promising. However, it has a computational cost. The per-step calculation of the Frobenius norm (`g.norm()`) and the dual norm (`torch.tensordot`), even if individually fast, adds up to a significant overhead when performed across all layers at every iteration. This highlights that an optimizer must be evaluated not just on its theoretical convergence rate in steps, but on its practical convergence rate per second. The additional FLOPs introduced by Arion's adaptivity were not "free," and in this case, the simpler and faster Muon update won the race.

**3. The Path Forward: Cheaper Proxies and Asynchronous Updates**
The challenge is to capture the benefits of adaptivity without paying the full computational price. A few avenues:

* **Cheaper Curvature Proxies:** Is `g.norm()` the most efficient proxy for local smoothness? Perhaps a less frequently updated or cheaper-to-calculate metric could provide a good enough signal.
* **Asynchronous Adaptivity:** Does the adaptive radius `t` need to be updated at every single step? It's possible that updating it every 5 or 10 steps would provide most of the benefit while amortizing the computational cost.
* **Hybrid Approaches:** Perhaps the optimal solution is a hybrid: use the fixed `sqrt(fan-out/fan-in)` scaling from Muon, but use a very simple, low-cost adaptive global multiplier based on the average gradient norm across the entire model, rather than a per-layer calculation.

Overall, this experiment demonstrated the trade-offs between theoretical experiments and practical performance, and that simplicity and low FLOP count are king. 

## References 
1. Muon. Keller Jordan blog. See: [https://kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon). 
2. Liu, J., Su, J., Yao, X., Jiang, Z., Lai, G., Du, Y., ... & Yang, Z. (2025). Muon is scalable for llm training. arXiv preprint arXiv:2502.16982.
3. NanoGPT Speedrun. See: [https://github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). 
4. Pethick, T., Xie, W., Antonakopoulos, K., Zhu, Z., Silveti-Falls, A., & Cevher, V. (2025). Training Deep Learning Models with Norm-Constrained LMOs. arXiv preprint arXiv:2502.07529.
5. Bernstein, J., & Newhouse, L. (2024). Modular duality in deep learning. arXiv preprint arXiv:2410.21265.
6. Riabinin, A., Shulgin, E., Gruntkowska, K., & Richtárik, P. (2025). Gluon: Making Muon & Scion Great Again!(Bridging Theory and Practice of LMO-based Optimizers for LLMs). arXiv preprint arXiv:2505.13416.
7. Bernstein, J., & Newhouse, L. (2024). Old optimizer, new norm: An anthology. arXiv preprint arXiv:2409.20325.
8. Muon and a Selective Survey on Steepest Descent in Riemannian and Non-Riemannian Manifolds. Franz Louis Cesista Blog. See: [https://leloykun.github.io/ponder/steepest-descent-non-riemannian](https://leloykun.github.io/ponder/steepest-descent-non-riemannian)
9. Yang, G., Simon, J. B., & Bernstein, J. (2023). A spectral condition for feature learning. arXiv preprint arXiv:2310.17813.
10. Deriving Muon. Jeremy Bernstein. See: [https://jeremybernste.in/writing/deriving-muon](https://jeremybernste.in/writing/deriving-muon)







