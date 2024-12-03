---
title: 'Enhancing Factual Accuracy in Large Language Models: Integrating Decoding Strategies and Model Steering'
date: 2024-12-02
permalink: /posts/2024/12/enhancing-factual-accuracy-in-large-language-models
tags:
  - artificial intelligence 
  - hallucinations 
  - large language models
  - interpretability 
---

The emergence of open-source Large Language Models (LLMs) like Llama has revolutionized natural language generation (NLG), making advanced conversational AI accessible to a broader audience [1]. Despite their impressive capabilities, these models often grapple with a significant challenge: factual hallucinations. Factual hallucinations occur when an AI model generates content that is unfaithful to the source material or cannot be verified against reliable data [2]. This issue is particularly concerning in critical and information-dense fields such as health, law, finance, and education, where misinformation can have catastrophic consequences [3][4].

This essay explores the integration of inference-time decoding strategies with model steering as a novel approach to enhance the factual accuracy of LLMs. By combining these two methods, we can potentially build adaptive systems capable of detecting and mitigating factual hallucinations. This integration holds promise for improving the reliability of AI-generated content in real-world applications, especially in domains where accuracy is paramount.

**Introduction**

The advent of LLMs has transformed artificial intelligence, enabling machines to generate human-like text and engage in complex dialogues. Models like Llama 3.1 and 3.2 have democratized access to state-of-the-art NLG technologies [1], fostering innovation across various industries. However, a persistent challenge remains: ensuring the factual accuracy of these models to minimize misinformation and hallucinations [2].

Factual hallucinations are particularly problematic in knowledge-grounded dialogue (KGD) systems, which combine tasks such as abstractive summarization, generative question answering (GQA), and dialogue generation [5]. In these systems, hallucinations not only undermine the model's faithfulness to source material but also pose significant risks by disseminating incorrect information [3, 4]. In critical domains like health, law, and finance (HLF), such inaccuracies can lead to life-altering decisions based on false premises. In educational contexts, they threaten the integrity of knowledge dissemination, potentially misinforming the next generation and eroding societal trust.

While private AI companies may have proprietary methods to mitigate hallucinations, there is a pressing need for open-source solutions that can be seamlessly integrated with models like Llama [1]. Developing such methods is crucial for businesses worldwide that rely on KGD systems in critical, information-dense domains and for delivering educational content responsibly.

**Challenges of Factual Hallucinations in LLMs**

Hallucinations in LLMs can arise from various factors, including poor training data quality, limitations in training objectives, architectural constraints, exposure bias, and inadequate decoding strategies. Despite advancements in training large models, completely eliminating hallucinations remains an innate limitation of current LLMs [6].

In critical domains, the stakes are high. For instance, a health chatbot providing incorrect medical advice can lead to severe health consequences. Similarly, legal or financial assistants generating erroneous information can result in substantial legal liabilities or financial losses. These risks highlight the urgent need to enhance the factual reliability of LLMs [3, 4].

**Existing Approaches to Mitigate Hallucinations**

Several strategies have been employed to reduce hallucinations in LLMs:

1. **Improving Data Quality**: Enhancing the quality and diversity of training data can help models learn more accurate representations of knowledge [1, 6, 7, 8].

2. **Fine-Tuning with Human/AI Feedback**: Methods like Reinforcement Learning from Human Feedback (RLHF) and AI feedback (RLAIF) allow models to learn from corrections, improving their factual accuracy over time [9-14].

3. **In-Context Learning and Specialized Prompting**: Providing models with contextually relevant information and carefully designed prompts can guide them toward generating more accurate responses [15-21].

4. **External Tools and Retrieval-Augmented Generation (RAG)**: Augmenting models with external knowledge sources or retrieval mechanisms can improve factual accuracy by grounding responses in verified information [22-27].

5. **Structured Outputs and Architectural Enhancements**: Constraining outputs to predefined structures and improving model architectures can reduce the likelihood of hallucinations [28-31].

6. **Inference-Time Decoding Strategies**: Adjusting the decoding process during inference to favor more reliable outputs has shown promise in enhancing factuality [32-45].

7. **Model Interventions and Steering**: Influencing the model's internal mechanisms to promote desired outcomes, such as factual accuracy, represents a growing area of research [46-51].

**Inference-Time Decoding Strategies**

Inference-time decoding is the process by which an AI model generates the next word in a sequence during text generation. Traditional decoding methods allow for a wide range of possibilities, increasing the risk of generating inaccurate or unverified information. By refining these strategies, it is possible to guide the model toward more factual outputs.

One approach is to truncate the unreliable tail of the probability distribution, sampling only from the top-k most probable tokens or those within the top-p portion of the probability mass [32, 42]. Dynamic adjustments to these parameters, informed by factors like token distribution entropy or the model's confidence levels, can further enhance output quality [34-37, 43, 44]. For instance, Entropix utilizes entropy-based scores to adjust sampling thresholds dynamically, aiming to reduce the likelihood of hallucinations without compromising the model's creativity [45, 52].

Other novel decoding strategies include COT-decoding [33], uncertainty-aware beam search [38], context-aware decoding [40], and methods like DoLa, which contrasts earlier and later layers for better sampling [41]. These techniques seek to balance the trade-off between generating diverse language and maintaining factual accuracy.

**Model Steering**

Model steering involves influencing the internal processes of an AI model to promote desired outcomes—in this case, factual accuracy. Previous work has shown that LLMs act as knowledge bases [53, 54] and have a general sense of what they know and what they don’t know [55, 56]. Furthermore, there exist knowledge neurons responsible for factual knowledge [57], and entropy and token frequency neurons that affect logit distributions [58, 59]. Therefore, hidden neurons can provide useful information about LLM uncertainty [60], and steering them can modulate it [61].

Techniques like DoLa and INSIDE utilize internal layer comparisons and covariance analyses to detect and correct potential hallucinations before the final output is produced [41, 48]. Additionally, methods such as Inference-Time Intervention (ITI) select attention heads with high probing accuracy for truthfulness and shift activations along these truth-correlated directions during inference [46].

Sparse autoencoders (SAEs) offer another avenue for model steering by learning sparse feature dictionaries that can represent meaningful directions within the model's activation space [62, 63]. Steering the model along these directions has shown early promise in modulating outputs toward desired attributes, such as factuality [64].

**Integrating Decoding Strategies with Model Steering**

Combining inference-time decoding strategies with model steering could be an interesting approach to reduce factual hallucinations. By simultaneously refining the output selection process and adjusting the model's internal mechanics, this strategy can potentially tackle some of the root causes of hallucinations.

This integrated approach involves:

- **Layer-Wise Analysis**: Examining the contributions of different layers within the model to identify where factual inaccuracies may originate (using probing or SAEs). Understanding the unique characteristics of each layer's hidden representations, both in MLP and self-attention blocks, can inform where and how to apply interventions effectively (steering). If certain layers and blocks are identified as being more important than others for factuality, these can be selectively intervened, and also be up-weighted during the decoding process.  

- **Score Integration**: Utilizing various entropy and agreement scores from methods like Entropix, EigenScore, and DoLa to inform decoding parameters. These scores can be tabulated at various layers and blocks, and help determine the optimal sampling thresholds. Different scores can be combined using an algorithm that weights these scores appropriately. 

- **Adaptive Interventions**: Developing algorithms that can automatically determine when to apply either intervention or combine them based on the content being generated and the model's internals. This adaptability is crucial for handling the diverse range of inputs and contexts that LLMs encounter in real-world applications.

**Applications in Critical Domains and Education**

Implementing these advanced strategies has significant implications for real-world applications, particularly in domains where accuracy is crucial.

**Healthcare**: In the health domain, AI assistants equipped with enhanced factuality mechanisms can provide reliable information to patients and professionals alike, improving outcomes and building trust in AI solutions. Accurate AI support can assist in symptom checking, patient education, and even preliminary diagnosis, provided that the information is verifiable and trustworthy.

**Legal and Financial Services**: Legal and financial professionals can leverage more reliable AI assistants for contract analysis, legal research, client interactions, and financial advising. Minimizing the risk of incorrect information is essential for compliance and maintaining professional standards.

**Education**: In educational contexts, reducing factual hallucinations ensures that students receive accurate information, supporting effective learning and preventing the spread of misinformation. As AI becomes more integrated into educational tools and platforms, maintaining high standards of factual accuracy is essential for fostering an informed and knowledgeable society.

**Societal Implications and Trust in AI**

Addressing factual hallucinations in LLMs is not just a technical challenge but a societal imperative. Misinformation inadvertently spread through AI interactions is a major concern, particularly when it involves knowledge that is considered a public good. Factual inaccuracies in critical domains can erode public trust in AI technologies and hinder their adoption [65]. Integrating decoding strategies with model steering could address this, and lead us toward more reliable and trustworthy AI. Enhancing the factual accuracy of LLMs paves the way for AI that not only communicates fluently but also informs responsibly, supporting better decision-making across various facets of society. The potential impact spans critical domains such as health, law, finance, and education, where the accuracy of information is paramount. As reliance on AI systems for knowledge acquisition and decision-making grows, the continued development and refinement of these integrated strategies will play a crucial role in shaping the future of AI, fostering greater trust and utility in technologies that are increasingly integral to our lives.


**References**

1. Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., ... & Ganapathy, R. (2024). The llama 3 herd of models. arXiv preprint arXiv:2407.21783.
2. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12), 1-38.
3. Guo, Z., Schlichtkrull, M., & Vlachos, A. (2022). A survey on automated fact-checking. Transactions of the Association for Computational Linguistics, 10, 178-206.
4. Wang, Y., Wang, M., Manzoor, M. A., Liu, F., Georgiev, G., Das, R. J., & Nakov, P. (2024). Factuality of large language models in the year 2024. arXiv preprint arXiv:2402.02420.
5. Xu, Y., Kong, D., Xu, D., Ji, Z., Pang, B., Fung, P., & Wu, Y. N. (2023, July). Diverse and faithful knowledge-grounded dialogue generation via sequential posterior inference. In International Conference on Machine Learning (pp. 38518-38534). PMLR.
6. Xu, Z., Jain, S., & Kankanhalli, M. (2024). Hallucination is inevitable: An innate limitation of large language models. arXiv preprint arXiv:2401.11817.
7. Li, Y., Bubeck, S., Eldan, R., Del Giorno, A., Gunasekar, S., & Lee, Y. T. (2023). Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463.
8. Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., ... & Sifre, L. (2022, June). Improving language models by retrieving from trillions of tokens. In International conference on machine learning (pp. 2206-2240). PMLR.
9. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35, 27730-27744.
10. Lin, Y., Tan, L., Lin, H., Zheng, Z., Pi, R., Zhang, J., ... & Zhang, T. (2023). Speciality vs generality: An empirical study on catastrophic forgetting in fine-tuning foundation models. arXiv preprint arXiv:2309.06256.
11. Elaraby, M., Lu, M., Dunn, J., Zhang, X., Wang, Y., Liu, S., ... & Wang, Y. (2023). Halo: Estimation and reduction of hallucinations in open-source weak large language models. arXiv preprint arXiv:2308.11764.
12. Zhang, H., Diao, S., Lin, Y., Fung, Y. R., Lian, Q., Wang, X., ... & Zhang, T. (2023). R-tuning: Teaching large language models to refuse unknown questions. arXiv preprint arXiv:2311.09677.
13. Tian, K., Mitchell, E., Yao, H., Manning, C. D., & Finn, C. (2023). Fine-tuning language models for factuality. arXiv preprint arXiv:2311.08401.
14. Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., ... & Kaplan, J. (2022). Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.
15. Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X., Celikyilmaz, A., & Weston, J. (2023). Chain-of-verification reduces hallucination in large language models. arXiv preprint arXiv:2309.11495.
16. Lei, D., Li, Y., Hu, M., Wang, M., Yun, V., Ching, E., & Kamal, E. (2023). Chain of natural language inference for reducing large language model ungrounded hallucinations. arXiv preprint arXiv:2310.03951.
17. Zheng, C., Li, L., Dong, Q., Fan, Y., Wu, Z., Xu, J., & Chang, B. (2023). Can we edit factual knowledge by in-context learning?. arXiv preprint arXiv:2305.12740.
18. Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325.
19. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.
20. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2024). Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36.
21. Yu, S., Bao, R., Bhatia, P., Kass-Hout, T., Zhou, J., & Xiao, C. (2024). Dynamic Uncertainty Ranking: Enhancing In-Context Learning for Long-Tail Knowledge in LLMs. arXiv preprint arXiv:2410.23605.
22. Gou, Z., Shao, Z., Gong, Y., Shen, Y., Yang, Y., Duan, N., & Chen, W. (2023). Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738.
23. Varshney, N., Yao, W., Zhang, H., Chen, J., & Yu, D. (2023). A stitch in time saves nine: Detecting and mitigating hallucinations of llms by validating low-confidence generation. arXiv preprint arXiv:2307.03987.
24. Kang, H., Ni, J., & Yao, H. (2023). Ever: Mitigating hallucination in large language models through real-time verification and rectification. arXiv preprint arXiv:2311.09114.
25. Gao, T., Yen, H., Yu, J., & Chen, D. (2023). Enabling large language models to generate text with citations. arXiv preprint arXiv:2305.14627.
26. Vu, T., Iyyer, M., Wang, X., Constant, N., Wei, J., Wei, J., ... & Luong, T. (2023). Freshllms: Refreshing large language models with search engine augmentation. arXiv preprint arXiv:2310.03214.
27. Gao, L., Dai, Z., Pasupat, P., Chen, A., Chaganty, A. T., Fan, Y., ... & Guu, K. (2022). Attributed text generation via post-hoc research and revision. arXiv preprint arXiv:2210.08726.
28. Rumiantsau, M., Vertsel, A., Hrytsuk, I., & Ballah, I. (2024). Beyond Fine-Tuning: Effective Strategies for Mitigating Hallucinations in Large Language Models for Data Analytics. arXiv preprint arXiv:2410.20024.
29. Balakrishnan, A., Rao, J., Upasani, K., White, M., & Subba, R. (2019). Constrained decoding for neural NLG from compositional representations in task-oriented dialogue. arXiv preprint arXiv:1906.07220.
30. Li, J., Consul, S., Zhou, E., Wong, J., Farooqui, N., Ye, Y., ... & Diamos, G. (2024). Banishing LLM hallucinations requires rethinking generalization. arXiv preprint arXiv:2406.17642.
31. Verma, S., Tran, K., Ali, Y., & Min, G. (2023). Reducing llm hallucinations using epistemic neural networks. arXiv preprint arXiv:2312.15576.
32. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751.
33. Wang, X., & Zhou, D. (2024). Chain-of-thought reasoning without prompting. arXiv preprint arXiv:2402.10200.
34. Hewitt, J., Manning, C. D., & Liang, P. (2022). Truncation sampling as language model desmoothing. arXiv preprint arXiv:2210.15191.
35. Basu, S., Ramachandran, G. S., Keskar, N. S., & Varshney, L. R. (2020). Mirostat: A neural text decoding algorithm that directly controls perplexity. arXiv preprint arXiv:2007.14966.
36. Lee, N., Ping, W., Xu, P., Patwary, M., Fung, P. N., Shoeybi, M., & Catanzaro, B. (2022). Factuality enhanced language models for open-ended text generation. Advances in Neural 37. Information Processing Systems, 35, 34586-34599.
37. Nguyen, M., Baker, A., Kirsch, A., & Neo, C. (2024). Min P Sampling: Balancing Creativity and Coherence at High Temperature. arXiv preprint arXiv:2407.01082.
38. Xiao, Y., & Wang, W. Y. (2021). On hallucination and predictive uncertainty in conditional language generation. arXiv preprint arXiv:2103.15025.
39. Rebuffel, C., Roberti, M., Soulier, L., Scoutheeten, G., Cancelliere, R., & Gallinari, P. (2022). Controlling hallucinations at word level in data-to-text generation. Data Mining and Knowledge Discovery, 1-37.
40. Shi, W., Han, X., Lewis, M., Tsvetkov, Y., Zettlemoyer, L., & Yih, S. W. T. (2023). Trusting your evidence: Hallucinate less with context-aware decoding. arXiv preprint arXiv:2305.14739.
41. Chuang, Y. S., Xie, Y., Luo, H., Kim, Y., Glass, J., & He, P. (2023). Dola: Decoding by contrasting layers improves factuality in large language models. arXiv preprint arXiv:2309.03883.
42. Fan, A., Lewis, M., & Dauphin, Y. (2018). Hierarchical neural story generation. arXiv preprint arXiv:1805.04833.
43. Meister, C., Pimentel, T., Wiher, G., & Cotterell, R. (2023). Locally typical sampling. Transactions of the Association for Computational Linguistics, 11, 102-121.
44. Ravfogel, S., Goldberg, Y., & Goldberger, J. (2023). Conformal nucleus sampling. arXiv preprint arXiv:2305.02633.
45. Xjdr, & doomslide. (2024). Entropix [Code repository]. GitHub. https://github.com/xjdr-alt/entropix.
46. Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2024). Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36.
47. Daheim, N., Dziri, N., Sachan, M., Gurevych, I., & Ponti, E. M. (2023). Elastic weight removal for faithful and abstractive dialogue generation. arXiv preprint arXiv:2303.17574.
48. Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., ... & Ye, J. (2024). INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection. arXiv preprint arXiv:2402.03744.
49. Yu, L., Cao, M., Cheung, J. C. K., & Dong, Y. (2024). Mechanisms of non-factual hallucinations in language models. arXiv preprint arXiv:2403.18167.
50. De Cao, N., Aziz, W., & Titov, I. (2021). Editing factual knowledge in language models. arXiv preprint arXiv:2104.08164.
51. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 35, 17359-17372.
52. Southbridge.ai. (2024). Entropixplained. https://southbridge-research.notion.site/Entropixplained-11e5fec70db18022b083d7d7b0e93505.
53. Aspillaga, C., Mendoza, M., & Soto, A. (2021). Inspecting the concept knowledge graph encoded by modern language models. arXiv preprint arXiv:2105.13471.
54. Petroni, F., Rocktäschel, T., Lewis, P., Bakhtin, A., Wu, Y., Miller, A. H., & Riedel, S. (2019). Language models as knowledge bases?. arXiv preprint arXiv:1909.01066.
55. Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., ... & Kaplan, J. (2022). Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.
56. Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., & Huang, X. (2023). Do Large Language Models Know What They Don't Know?. arXiv preprint arXiv:2305.18153.
57. Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2021). Knowledge neurons in pretrained transformers. arXiv preprint arXiv:2104.08696.
58. Katz, S., & Belinkov, Y. (2023). VISIT: Visualizing and interpreting the semantic information flow of transformers. arXiv preprint arXiv:2305.13417.
59. Stolfo, A., Wu, B., Gurnee, W., Belinkov, Y., Song, X., Sachan, M., & Nanda, N. (2024). Confidence regulation neurons in language models. arXiv preprint arXiv:2406.16254.
60. Liu, L., Pan, Y., Li, X., & Chen, G. (2024). Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach. arXiv preprint arXiv:2404.15993.
61. Rahn, N., D'Oro, P., & Bellemare, M. G. (2024). Controlling Large Language Model Agents with Entropic Activation Steering. arXiv preprint arXiv:2406.00244.
62. Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., ... Olah, C. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread.  https://transformer-circuits.pub/2023/monosemantic-features.
63. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600.
64. Chalnev, S., Siu, M., & Conmy, A. (2024). Improving Steering Vectors by Targeting Sparse Autoencoder Features. arXiv preprint arXiv:2411.02193.
65. Evidently AI. (2024). LLM Hallucination Examples. Retrieved from https://www.evidentlyai.com/blog/llm-hallucination-examples.



