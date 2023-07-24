---
title: 'Developments in Machine Learning for Antibody Design'
date: 2023-07-24
permalink: /posts/2023/07/developments-in-ml-for-antibody-design
tags:
  - antibody design 
  - artificial intelligence 
  - biologics 
---

Protein structure and sequence modeling has seen a fresh wave of resurgence in the last couple of years owing to some interesting developments in machine learning (ML) and deep learning (DL) based techniques. These techniques appear in a variety of flavours including using Equivariant neural network modules to respect the structural properties of 3D macromolecules, deeper networks that can benefit from the increased available experimental structures, powerful node-to-node relationship learners like transformers, and masked language modeling on the protein sequence space to learn evolutionary information. While structure prediction methods like AlphaFold (AF) and RosettaFold (RF) have become ubiquitious in computational structural biology, there remain challenges to be tackled on multiple fronts, where ML will play an important role. From here on I intend to use ML as a singular term for the group of techniques that learn parameters of models from data using gradient descent, irrespective of the architecture or the size of the models employed. 

Proteins come in many shapes and forms and perform a multitude of functions in the body, and antibodies are one such class of proteins. Antibodies are an important group of proteins that are key players in the workings of immune system. When a foreign protein antigen enters the body, it is broken down into peptide fragments which attach to MHC molecules in the antigen presenting cells. These peptide fragments are then presented to the B-Cells.  The B-Cells where the B-Cell receptors bind to these peptides, create antibodies that bind to the antigen in an epitope specific way. This process is repeated to eventually obtain antibodies that have high affinity and specificity to the antigen in a process called somatic hypermutation. Exactly how somatic hypermutation is able to generate high affinity antibodies to any potential protein antigen is still not fully known, although there are theories [][]. The ability to do this computationally, that is given any protein target, designing antibodies that can bind to it with high affinity, is of extremely high therapeutic relevance. The use of antibodies as therapeutics is relatively under-explored with less than 150 FDA approved antibodies to date.  

While general protein structure prediction from sequence has reached very low RMSD with respect to experimental structures, antibody structure prediction is more challenging due to the presence of highly diverse and flexible loops called complimentarity determining regions (CDRs). Antibodies are Y-shaped molecules with a fixed region called fragment crystallizable (FC) region and two arms which are called fragment antibody binding (FAb) regions. The FAbs are comprised of a constant domain and a variable domain, where the constant domain is relatively conserved across different antibodies and the variable domain shows higher variablity. Most of the variability in the variable domain comes from the CDRs at the tips of these domains that form a dominant part of the paratope that binds to the antigen epitope. Each arm is also segregated into two separate chains, the heavy chain and the light chain, with CDR-H1, H2, H3 in the heavy chain and CDR-L1, L2, L3 in the light chain. Among the 6 CDR regions, CDR-H3 plays the most important role in binding, is the longest and also the most diverse.           


<a href="https://www.popsci.com/article/technology/surprising-science-behind-movie-interstellar/"><u>The science behind Interstellar</u></a>, 

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/data_poverty.png?raw=true">
</p>



