---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}
 
I am a machine learning engineer using machine learning/deep learning to understand and design biomolecules.

Education
======
* Ph.D. in Electrical and Computer Engineering (<span style="color:#2C4381">Thesis: Representation Learning Strategies for the Epigenome and Chromatin Structure using Recurrent Neural Models)</span>, The University of British Columbia, Vancouver, 2023 
* M.A.Sc. in Electrical and Computer Engineering, The University of British Columbia, Vancouver, 2018
* B.Tech. in Electronics and Communication Enginnering, National Institute of Technology Karnataka, India, 2017 

Work experience
======
* ## Machine Learning Engineer, <a href="https://www.gandeeva.com/"><u>Gandeeva Therapeutics</u></a> <span style="color:#2C4381">(Feb 2023 - May 2023)</span> 
  * Integrated molecular dynamics based conformation search with deep learning based protein design for protein affinity maturation. The designed protein sequences achieved SOTA success rate when tested in the lab. Gandeeva will use this platform for their future protein affinity maturation campaigns
  * Tested interface scoring tools from CASP15 to drive in-house ppi prediction, which will significantly improve Gandeeva’s target discovery efforts
  * Tested proof-of-concept target-potential binding partner interface recovery by using protein design to suggest favourable mutations on the potential binding partner. This will help Gandeeva recover weak but therapeutically relevant interfaces
  * Provided critical feedback on the in-house active learning+molecular docking workflow for virtual screening
  * **Manager**: Abhishek Mukhopadhyay

* ## Ph.D. Candidate, The University of British Columbia <span style="color:#2C4381">(Jan 2019 - Feb 2023)</span> 
  * My thesis studied learning strategies that can design representations for the <a href="https://kevinbdsouza.github.io/publications/epilstm"><u>epigenome</u></a> and <a href="https://kevinbdsouza.github.io/publications/hiclstm"><u>chromatin structure</u></a>
  * These representations are useful for a variety of tasks like pan-celltype element identification, novel element detection, transfer learning to unseen cell types, inference of 3D chromatin structure, and in-silico modifications
  * The software for my research is written in Python using PyTorch and CUDA. The software packages resulting from this work are available on <a href="https://github.com/kevinbdsouza"><u>GitHub</u></a>
  * **Supervisor**: Prof. Maxwell Libbrecht (<a href="https://www.libbrechtlab.com"><u>Libbrecht lab</u></a>) and Prof. Vijay Bhargava

* ## Machine Learning Intern, <a href="https://www.gandeeva.com/"><u>Gandeeva Therapeutics</u></a> <span style="color:#2C4381">(Jun 2022 - Dec 2022)</span> 
  * I designed antibodies using sequence and structure based Machine Learning, protein folding tools, and rosetta energy metrics
  * **Manager**: Abhishek Mukhopadhyay

* ## Machine Learning Intern, <a href="https://www.skycope.com/"><u>Skycope Technologies</u></a> <span style="color:#2C4381"> (May 2018 - Sep 2018)</span>
  * Built Skycope's data and machine learning infrastructure 
  * Set up Skycope's database server and backend MySQL software for data extraction and manipulation 
  * Modified Faster-RCNN, an existing object detection framework, to successfully <a href="https://kevinbdsouza.github.io/publications/frcnn"><u>detect and locate drone signals in the spectrogram</u></a>
  * Integrated ML software into Skycope's existing software stack 
  * Skycope is now using this as their flagship service in their automatic drone detection software
  * **Supervisor**: Hamidreza Boostanimehr

* ## M.A.Sc. Research Assistant, The University of British Columbia <span style="color:#2C4381"> (Sep 2017 - Dec 2018)</span>
  * Adopted deep learning frameworks for signal detection and developed hybrid precoding schemes for sequential data
  * Used Python, PyTorch, and CUDA for deep learning
  * **Supervisor**: Prof. Vijay Bhargava

  
Skills
======
* Computational Biology
* Machine Learning 
  * Deep Learning, Transformers, Active Learning, Generative Modeling, Language Models, Sequential Models, Representation Learning, Statistics, Regression, Gradient Boosted Trees, MCMC
* Coding 
  * Python, C, C++, Linux, Git, MySQL
* ML tools 
  * PyTorch, GCP, CUDA, AWS, XGBoost, Docker
* Biology 
  * Protein Structure, Protein Design, Antibody Engineering, Molecular Dynamics, FEP, Docking, Chromatin Structure, Epigenomics, NGS, Genomics, Transcriptomics, Multiomics, RNA-seq, ChIP-seq, Hi-C, scRNA-seq
* Softwares
  * OpenMM, Pymol, Chimera, Gromacs, Rosetta, AlphaFold, Schrodinger
* Data Science 
* Bioinformatics
* Software Development 



Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Service 
======
* Joint-Secretary at Autism Society of Udupi (ASU)
  * ASU is a non-profit organisation in Udupi, India, that aims to create awareness about Autism among parents, teachers, health professionals, students, general public and all the stakeholders so that early diagnosis and early intervention could give the child maximum benefits
  * Currently we are working alongside primary schools to educate teachers about autism. Partnering with various organisations working towards similar causes 
* Mentor at Climate Hub, UBC
  * The UBC Climate Hub aims to connect university and community stakeholders to take bold climate action for a just and equitable future
  * I worked with undergraduate students to find interdisciplinary approaches that contribute to climate awareness
* Mentor at iGEM, UBC
  * The International Genetically Engineered Machines (iGEM) Competition aims to produce organisms with unusual properties to tackle modern problems using synthetic biology
  * I worked with undergraduate students to design a transcription-based biosensor that detects levels of saxitoxin, a toxin responsible for paralytic shellfish poisoning in humans
* Volunteer at Lets Talk Science and Geneskool, Genome BC
  * Mentored high school students to inculcate in them aptitude and curiosity towards science
  * Conducted workshops on the basics of genetics, traits and phenotypes for elementary grade students

Projects
======
  <ul>{% for post in site.projects reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
