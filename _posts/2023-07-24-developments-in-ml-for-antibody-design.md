---
title: 'Developments in Machine Learning for Antibody Design'
date: 2023-07-24
permalink: /posts/2023/07/developments-in-ml-for-antibody-design
tags:
  - antibody design 
  - artificial intelligence 
  - biologics 
---

Protein structure and sequence modeling has seen a fresh wave of resurgence in the last couple of years owing to some interesting developments in machine learning (ML) and deep learning (DL) based techniques. These techniques appear in a variety of flavours including using Equivariant neural network modules to respect the structural properties of 3D macromolecules, deeper networks that can benefit from the increased available experimental structures, powerful node-to-node relationship learners like transformers, and masked language modeling on the protein sequence space to learn evolutionary information. While structure prediction methods like AlphaFold (AF) and RosettaFold (RF) have become ubiquitious in computational structural biology, there remain challenges to be tackled on multiple fronts, where ML will play an important role. 

From here on I intend to use ML as a singular term for the group of techniques that learn parameters of models from data using gradient descent, irrespective of the architecture or the size of the models employed. 

Proteins come in many shapes and forms and perform a multitude of functions in the body, and antibodies are one such class of proteins. Antibodies are an important group of proteins that are key players in the workings of immune system. When a foreign protein antigen enters the body, it is broken down into peptide fragments which attach to MHC molecules in the antigen presenting cells. These peptide fragments are then presented to the B-cells.  The B-cells where the B-cell receptors bind to these peptides, create antibodies that bind to the antigen in an epitope specific way. This process is repeated to eventually obtain antibodies that have high affinity and specificity to the antigen in a process called somatic hypermutation. Exactly how somatic hypermutation is able to generate high affinity antibodies to any potential protein antigen is still not fully known, although there are theories [][]. The ability to do this computationally, that is given any protein target, designing antibodies that can bind to it with high affinity, is of extremely high therapeutic relevance. The use of antibodies as therapeutics is relatively under-explored with less than 150 FDA approved antibodies to date.  

While general protein structure prediction from sequence has reached very low RMSD with respect to experimental structures, antibody structure prediction is more challenging due to the presence of highly diverse and flexible loops called complimentarity determining regions (CDRs). Antibodies are Y-shaped molecules with a fixed region called fragment crystallizable (FC) region and two arms which are called fragment antibody binding (FAb) regions. The FAbs are comprised of a constant domain and a variable domain, where the constant domain is relatively conserved across different antibodies and the variable domain shows higher variablity. Most of the variability in the variable domain comes from the CDRs at the tips of these domains that form a dominant part of the paratope that binds to the antigen epitope. Each arm is also segregated into two separate chains, the heavy chain and the light chain, with CDR-H1, H2, H3 in the heavy chain and CDR-L1, L2, L3 in the light chain. Among the 6 CDR regions, CDR-H3 plays the most important role in binding, is the longest and also the most diverse.   

Given the peculiarities of antibodies, designing them computationally to bind to a target protein has been challenging. I categorize this broader problem into three categories that differ in their available inputs and end goal. 

* In-silico affinity maturation
* CDR infilling
* De novo design    

In-silico affinity maturation aims to enhance the binding affinity of antibodies that are known to bind to the target. This is primarily achieved by a framework similar to the one shown in Fig. 1. Representations of B-cell receptor repertoires are first learned using either antibody language models or other representation learning techniques. Antibody language models are increasingly becoming powerful at creating useful representations which can then be used for downstream tasks (examples include IgLM [], AbLang []). These representations are then used to train a regression model that predicts the target binding affinity of sequences by using measured experimental data as the ground truth. The trained regression model is then used in an in-silico directed evolution campaign to generate new sequences. The in-silico directed evolution process can take many forms including active learning or bayesian optimization. RESP AI [], the geometric deep learning (GDL) model by Peng [], and the language models assisted bayesian optimization framework by Walsh [] come under this first category.        

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/aff_mat.png?raw=true" width="500"/>
</p>
<p align="center">
<em>Figure 1. Framework for in-silico affinity maturation.</em>
</p>

The second category pertains to designing alternate CDR regions that bind to an antigen with greater affinity given an existing antibody-antigen complex structure. This is achieved by a framework similar to the one shown in Fig. 2. Existing antibody-antigen complexes are used ad training data to train a model that can predict either the sequence, or the backbone structure, or both simultaneously. Some of the models in this category include MEAN [], the model from Absci [], AntiDesigner [], DiffAb [], Fragment-based method by Sormanni [], C-RefineGNN [], and AbODE []. Some of these methods use diffusion models, some work with partial differential equations (PDEs), and most methods work on top of graphs. CDR evolution in the sequence space is also another class of methods in this category, shown in Fig. 3. These methods use antibody language model log probabilities to chose fitter CDR regions in the sequence space. Examples inlcude methods like IgLM [], AbMAP [], AbLang [], ESM (Kim) [], ReprogBert []. 
 
<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/cdr_infill.png?raw=true" width="400"/>
</p>
<p align="center">
<em>Figure 2. CDR sequence/structure infilling scenario.</em>
</p>

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/cdr_infill_2.png?raw=true" width="400"/>
</p>
<p align="center">
<em>Figure 2. CDR sequence evolution scenario.</em>
</p>

The third category consists of designing the entire antibody from scratch such that it can bind to a given target. While the first category is epitope agnostic, the second and the third categories are epitope specific, that is, the antibody needs to be designed such that it binds to a particular epitope of interest on the antigen. 

In progress. 





