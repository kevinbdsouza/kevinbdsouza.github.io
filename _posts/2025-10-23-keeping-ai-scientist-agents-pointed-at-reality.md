---
title: 'Keeping AI Scientist Agents Pointed at Reality'
category: 'AI'
date: 2025-10-23
permalink: /posts/2025/10/keeping-ai-scientist-agents-pointed-at-reality
tags:
  - AI scientist agents
  - AI safety
  - scientific discovery
  - scalable oversight
---

I am fairly confident that the oversight problem around AI scientist agents will become important in the near future, although I am much less confident about which specific oversight mechanisms will survive contact with stronger agents and the strange incentives of scientific institutions.

AI scientist agents are often judged by their ability to produce something that resembles a research paper. However, scientific discovery is more of a feedback process where ideas are repeatedly forced into contact with all the awkward resistance of the world. This resistance can come in many forms depending on the domain, whether it requires conducting experiments in the world as in biology/material science, running high-fidelity process-based simulations as in climate, or verifying against a formal checker as in mathematics.

The first useful AI scientist agents may look less like autonomous geniuses and more like tireless collaborators embedded inside this feedback loop, proposing hypotheses, choosing what to test, interpreting the results, and then deciding what should be tried next. This loop is where the oversight problem exists, because an agent can make many small decisions that shape the direction of discovery long before there is any result to evaluate. I hope people largely agree that a system that suggests the next molecule to synthesize, the next genotype to screen, or the next simulation to run is already participating in science in a meaningful way. The question is whether the path that led there was truth-seeking and robust.

## Discovery happens through the loop

I propose that the scientific object we should care about is the whole loop from hypothesis to experiment to feedback to revised hypothesis. Some of the common failure modes in such a loop can be an AI agent choosing a convenient proxy, generalizing from a narrow simulation, or returning to hypotheses that look promising under early feedback while gradually drifting away from the real target.

This would be quite familiar for us from human science, where a lab can spend months following a weak signal because the first few experiments were exciting, or a field can spend years optimizing a measurement that eventually turns out to be a poor stand-in for the thing everyone cared about. AI could make this pattern faster and wider, because it can generate more hypotheses and more experimental plans than humans can carefully verify.

The danger is therefore that the research process can become subtly misdirected. The agent may find something that looks like a discovery and the humans around it may feel that the loop is working because each step has a plausible local rationale. Scalable oversight for such a system involves deciding which hypothesis deserves the scarce experimental bandwidth and which proxies are good enough.

## Useful half-discoveries

The most difficult AI science failures may come from outputs that are partially useful, because fully absurd outputs are easier to discard and genuinely correct discoveries are easier to celebrate. For example, a candidate molecule may work in a narrow assay while failing under a more realistic model or a microbial consortium may look promising under one stress condition while collapsing outside it.

The phrase I keep coming back to is validation debt, by which I mean the gap between how promising a scientific direction looks and how much real checking has actually happened. AI scientist agents can accumulate validation debt very quickly, especially when early feedback is cheap and the decisive experiment is slow and expensive.

A lab may be able to test five candidates properly while an agent can propose five hundred. This has always been a problem in computational science as well, where the top of the funnel is large. However, this problem will further exacerbate when natural language agents interface with computational workflows and discovery systems become semi-automated.

## The capability gap inside scientific work

Scalable oversight is described as a weaker overseer checking a stronger system, and in scientific discovery this gap has many layers because the world itself is part of the evaluation process. The generator may have more model capacity, more tool calls, more context, and more attempts, while the human overseer may have deeper scientific taste but less time and less visibility into the full search process.

Some examples of judgement come to mind easily. A principal investigator might sense that a proposed mechanism is convenient while lacking the bandwidth to check every analysis step. A model reviewer might catch unsupported citations while missing the way an assay has become a proxy target. A technician might understand the fragility of a measurement in a way that no agent sees in the formal protocol. The oversight system has to combine these partial forms of judgement without assuming that any one of them sees the whole picture.

The gap can also be informational. For instance, if the overseer only sees the final recommendation, they cannot know how many failed branches were hidden by the agent's search process. If they see the polished explanation without the experimental history, they may mistake a selected version for a naturally emerging conclusion. The gap can be temporal as well, as some scientific claims reveal their truth only after slow feedback. An agent can recommend a promising direction today, a lab can spend months following it, and only later does it become clear that the loop was optimizing a convenient proxy.

This is why oversight for AI scientist agents has to be process-aware. The question is whether the whole discovery loop is being monitored in a way that preserves uncertainty, records failed paths, watches proxy drift, and escalates expensive claims quickly.

## Agent learns what the lab rewards

A closed-loop scientist agent will learn from experimental feedback, and this is exactly what makes it powerful. However, the same learning process also creates one of the core oversight risks, because the agent can become adapted to the lab's taste and the review procedures that decide whether its proposals are accepted. If the lab mostly rewards early assay signals, the agent may learn to produce candidates that look good under that assay. Similarly, if the computational pipeline relies heavily on a simulator, the agent may gradually learn the simulator's quirks.

To internalise this, we do not require any exotic story about deception. Search is already enough. When a system repeatedly generates candidates and keeps the ones that pass a filter, it will tend to find regions of the search space where the filter is easier to satisfy. That makes optimization pressure the real test of an oversight method. A review protocol that catches the first naive mistake may still collapse once the generator has learned to satisfy it, because the surviving errors will move into parts of the process that the review protocol cannot easily see.

In scientific discovery, those hidden parts might be any among the proxy, the experimental design, or the choice of failed runs to remember. The output can become more careful on the surface while the underlying search process becomes better at exploiting the available checks.

## Real experiments can still mislead the loop

There is a comforting thought that laboratory feedback will eventually discipline AI scientist agents, because nature pushes back in a way that language models cannot simply talk around. This is partly true, although the timescale and quality of the feedback matter enormously. Experiments often speak through narrow channels. For instance, a cheap assay may measure something related to the real target and an early phenotype may be easier to observe than the property that matters downstream. The agent can receive real feedback and still learn the wrong lesson if the measurement channel is misaligned with the scientific objective.

This is especially important in closed-loop discovery because the easiest feedback is often the feedback that gets optimized first. If a fast screen is cheap and mechanistic validation is slow, the agent may become excellent at producing fast-screen winners. If a simulation is cheap and field evidence is slow, the agent may become excellent at producing simulation winners. The gap between those winners and real discoveries is where many failures will live.

I feel good human scientists often understand this through tacit experience. They know which assays are treacherous, which controls matter, and which effects are usually artefacts. AI oversight needs to preserve that kind of judgement while also building machinery that can scale beyond one person.

## Oversight theatre in the laboratory

One of the failure modes I worry about is oversight theatre, where a review process creates the feeling of epistemic seriousness while leaving the important weaknesses untouched. In AI-driven science this can happen very naturally, because an agent can produce all the visible signs of careful scientific practice.

It can write a cautious experimental plan, add controls, cite relevant work, discuss limitations, and describe uncertainty in a tone that feels responsible. Those behaviours are useful, however they can also become more of a literary style, which makes poor evidence feel safer than it is. This probably matters more when the output influences real world experiments. A convincing explanation can lead a lab to synthesize a compound, start a field trial, or spend months on a mechanistic path that should have been treated as much more uncertain.

I feel a good oversight process has to ask the following questions.

- Whether the next experiment really distinguishes between competing hypotheses.
- Whether the proposed control would expose the relevant failure.
- Whether the proxy is still tied to the real objective.
- Whether the agent is reducing uncertainty.

Agent explanations are useful for inspection, however the strongest evidence will come from more concrete things such as a replication from a different starting point or a decisive experimental control.

## Many reviewers can share one blind spot

Multi-agent review feels appealing in this setting because it resembles peer review, and peer review is the social technology that science already uses to turn judgement into more reliable knowledge. I feel the key question here is whether the reviewers bring genuinely different evidence and failure modes. Several AI reviewers may appear independent while drawing on similar training data and similar preferences for plausible explanations. Therefore, a panel of such agents can produce a convincing consensus around a weak claim, especially when the original proposal already has the shape of scientific correctness.

Of course human science has the same problem at a slower speed. Fields can share assumptions, reward familiar methods, and treat certain proxies as respectable because the community has grown used to them. AI systems trained on the scientific literature may inherit these habits and then amplify them through scale.

The useful distinction here is between more opinions and more independent evidence. A second model reading the same action history adds a little information, whereas a replication agent starting from the raw data adds more. A domain expert questioning the construct being measured adds something different again. A decisive experiment that separates two causal stories adds the kind of evidence that can potentially change belief.

It seems that the future oversight stack should probably look like an evidence portfolio, with model review, code execution, experimental design checks, replication, domain expertise, and adversarial stress tests each playing a distinct role. The aim will be to make correlated blind spots visible before they become consensus during verification.

## Science of oversight

A mature science of oversight should tell us where a method works and where it degrades. Here are some examples to illustrate what I mean.

An oversight protocol may be reliable for choosing candidates for cheap proxy screening while being much weaker for deciding which candidates deserve expensive validation. A system may be useful for hypothesis generation while remaining unqualified to certify a mechanism. A model may help with literature-grounded reasoning in one domain while becoming dangerously persuasive in another domain where the validation channel is slower and weaker.

Therefore, a safety framework for an AI scientist agent should say which scientific decisions the system is allowed to influence and what evidence supports that permission. For instance, a research lab should be able to say that the agent can rank candidates for early screening under a defined protocol, that proxy drift is actively monitored, and that expensive validation decisions need human sign-off with access to the full search history.

I agree that this may sound bureaucratic, however, serious science has always depended on boring structures that slow down enthusiasm long enough for reality to answer. Statistical standards, peer review, replication norms, and safety protocols are all ways of protecting truth from wishful momentum.

## The future as epistemic routing

My rough picture is that AI science becomes a routing problem as much as a generation problem. Agents will generate ideas, while the scarce institutional resource will be deciding which branches of the discovery process deserve deeper contact with reality. The decisive oversight moments will occur throughout the loop in most domains.

The optimistic version is truly exciting. AI agents could make science more systematic by exploring more alternatives, remembering failed paths, and helping labs spend experimental budgets on better-chosen candidates. The pessimistic version is an epistemic swamp with excellent tooling, where human reviewers become throughput managers and the literature fills with discovery-shaped objects that can't be trusted.

Both futures seem available from where we are standing, and the difference may depend on whether verification improves alongside generation.

## The crux for AI science

If AI makes generation dramatically cheaper while leaving real validation almost as expensive as before, science may drown in plausible discovery-shaped objects. If AI also makes verification cheaper, the picture becomes much better. Agents could design stronger controls, search for disconfirming evidence, and allocate expensive experiments more intelligently. In that world, AI scientist agents become part of a stronger epistemic immune system.

It seems that different fields will land in different places. Mathematics and software have relatively strong validators. Biology has powerful feedback loops but often slow and messy grounding. Climate science can use simulation and observation while still struggling with long horizons and contested proxies. Materials science may move quickly where synthesis and characterization become automated, although real-world performance will still resist easy confirmation.

I believe the key is to remember that the real object in AI for scientific discovery is a family of discovery loops, each with its own incentives and failure modes. The AI scientist agents will be arriving through the lab notebook, the simulation pipeline, the assay queue, the codebase, the literature search, and the experimental planner. The question is whether we can build oversight systems that understand the loop deeply enough to keep discovery pointed at reality.
