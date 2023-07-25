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

Proteins come in many shapes and forms and perform a multitude of functions in the body, and antibodies are one such class of proteins. Antibodies are an important group of proteins that are key players in the workings of immune system and require B cells and T cells for their production. Foreign protein antigens that enter the body activate the B-cells in a T cell-dependent manner. Each B cell has multitudes of identical B cell receptors (BCRs) expressed on the surface that selectively bind to a particular epitope of the antigen. The protein antigen that binds to a BCR is taken up into the B cell through receptor-mediated endocytosis, degraded, and presented to Helper T cells as peptide fragments in complex with MHC-II molecules on the cell membrane. Helper T cells recognize and bind these MHC-II-peptide complexes through their T cell receptors (TCR). After TCR-MHC-II-peptide binding, T cells express certain surface proteins that are necessary for B cell activation. Activated B cells result in both short-lived variants that produce weak antibodies and long-lived variants that produce high affinity antibodies by a process called somatic hypermutation (SHM). SHM is a process in which point mutations accumulate in the variable domains of the antibody, particularly in the hypervariable regions, of both the heavy and light chains, at rates million-fold higher than the background mutation rates observed in other genes. While SHM does not distinguish between favorable and unfavorable mutations, the selection process selects for B cells that produce the highest-affinity antibodies. The ability to do this computationally, that is given any protein target, designing antibodies that can bind to it with high affinity (without multiple rounds of extensive mutations and selection), is of extremely high therapeutic relevance. The use of antibodies as therapeutics is relatively under-explored with less than 105 FDA approved antibodies to date.  

While general protein structure prediction from sequence has reached low RMSD with respect to experimental structures, antibody structure prediction is more challenging due to the presence of highly diverse and flexible loops called complimentarity determining regions (CDRs). Antibodies are Y-shaped molecules with a fixed region called fragment crystallizable (Fc) region and two arms which are called fragment antibody binding (Fab) regions. A variable number of intrachain disulphide bonds hold the Fc and Fab together, where the number varies depending on the type of antibody. The Fabs are comprised of a constant domain and a variable domain, where the constant domain is relatively conserved across different antibodies and the variable domain shows higher variablity. Most of the variability in the variable domain comes from the CDRs at the tips of these domains that form a dominant part of the paratope that binds to the antigen epitope. Each arm is also segregated into two separate chains, the heavy chain and the light chain, with CDR-H1, H2, H3 in the heavy chain and CDR-L1, L2, L3 in the light chain. Among the 6 CDR regions, CDR-H3 plays the most important role in binding, is the longest and also the most diverse.   

Given the peculiarities of antibodies, designing them computationally to bind to a target protein has been challenging. I categorize this broader problem into three categories that differ in their available inputs and end goal. 

* In-silico affinity maturation
* CDR infilling
* De novo design    

In-silico affinity maturation aims to enhance the binding affinity of antibodies that are known to bind to the target. This is primarily achieved by a framework similar to the one shown in Fig. 1. Representations of B-cell receptor repertoires are first learned using either antibody language models or other representation learning techniques. Antibody language models are increasingly becoming powerful at creating useful representations which can then be used for downstream tasks (examples include IgLM [], AbLang []). These representations are then used to train a regression model that predicts the target binding affinity of sequences by using measured experimental data as the ground truth. The trained regression model is then used in an in-silico directed evolution campaign to generate new sequences. The in-silico directed evolution process can take many forms including active learning or bayesian optimization. RESP AI [], the geometric deep learning (GDL) model by Peng [], and the language models assisted bayesian optimization framework by Walsh [] come under this first category.        

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/aff_mat.png?raw=true" width="500"/>
</p>
<p align="center">
<em> <font size="2"> Figure 1. Framework for in-silico affinity maturation.</font> </em>
</p>

The second category pertains to designing alternate CDR regions that bind to an antigen with greater affinity given an existing antibody-antigen complex structure. This is achieved by a framework similar to the one shown in Fig. 2. Existing antibody-antigen complexes are used ad training data to train a model that can predict either the sequence, or the backbone structure, or both simultaneously. The methods in this category include MEAN [], the model by Absci [], AntiDesigner [], DiffAb [], Fragment-based method by Sormanni [], C-RefineGNN [], and AbODE []. Some of these methods use diffusion models, some work with partial differential equations (PDEs), and most methods work on top of graphs. CDR evolution in the sequence space is also another class of methods in this category, shown in Fig. 3. These methods use antibody language model log probabilities to chose fitter CDR regions in the sequence space. Examples inlcude methods like IgLM [], AbMAP [], AbLang [], ESM based method by Kim [], and ReprogBert []. 
 
<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/cdr_infill.png?raw=true" width="400"/>
</p>
<p align="center">
<em> <font size="2"> Figure 2. CDR sequence/structure infilling scenario. </font> </em>
</p>

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/cdr_infill_2.png?raw=true" width="400"/>
</p>
<p align="center">
<em> <font size="2"> Figure 3. CDR sequence evolution scenario. </font> </em>
</p>

The third category consists of designing the entire antibody from scratch such that it can bind to a given target. While the first category is epitope agnostic, the second and the third categories are epitope specific, that is, the antibody needs to be designed such that it binds to a particular epitope of interest on the antigen. The problem of designing the antibody can be broken down into two subproblems of designing the framework region (Fab minus the hypervariable CDR regions) and designing the CDR regions (The Fc region is same for a given class of antibody in the species and both arms of the antibody can be identical). 

The framework of the antibody can be designed one of three ways. First, use existing framework strcutures from databases like SAbDab. Second, use antibody structure prediction methods to predict the structures of the framework regions in the antibody sequence databases like the observed antibody space (OAS), and third, de novo design the whole framework structure. There are methods that go the third route and try to either diffuse the framework stucture like Ig-VAE [] and PF-ABGEN [] (Fig. 4), or design it using cross-Beta motif rules such as the method by Marcos []. Designing a de novo framework is often not necessary because of the vast available sequence repertoires and good antibody structure prediction results for the framework region. Therefore, using the OAS in conjunction with antibody structure prediction methods is a good start given the small number of eperimentally available framework structures.    

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/diffuse.png?raw=true" width="200"/>
</p>
<p align="center">
<em> <font size="2"> Figure 4. De novo diffusion of antibody frameworks. </font> </em>
</p>


In progress. 





