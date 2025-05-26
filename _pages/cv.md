---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}
 
Interested in AI, its applications to science, and its relationship with society.

I'm a postdoctoral fellow at the University of Waterloo working at the intersection of AI and science. Previosuly, I was a ML engineer at <a href="https://www.gandeeva.com/"><u>Gandeeva Therapeutics</u></a>, where I designed proteins and <a href="https://kevinbdsouza.github.io/posts/2023/07/developments-in-ml-for-antibody-design"><u>antibodies</u></a>. 

For my Ph.D., I worked with the computational biology group at Simon Fraser University, while also being funded by UBC. At the <a href="https://www.libbrechtlab.com"><u>Libbrecht lab</u></a>, I designed representation learning strategies for <a href="https://kevinbdsouza.github.io/publications/hiclstm"><u>chromatin structure</u></a> and the <a href="https://kevinbdsouza.github.io/publications/epilstm"><u>epigenome</u></a>. Before my Ph.D., I completed my M.A.Sc while working with the information theory group at UBC. During this time, I interned with Skycope Technologies to build their <a href="https://kevinbdsouza.github.io/publications/frcnn"><u>automatic drone detection software</u></a>. I also have experience using machine learning in a variety of settings like <a href="https://kevinbdsouza.github.io/projects/privateml"><u>privacy preservation</u></a>, and <a href="https://kevinbdsouza.github.io/projects/vaelm"><u>variational language models</u></a>.

Education
======
* Ph.D. in Electrical and Computer Engineering (<span style="color:#2C4381">Thesis: Representation Learning Strategies for the Epigenome and Chromatin Structure using Recurrent Neural Models)</span>, The University of British Columbia, Vancouver, 2023 
* M.A.Sc. in Electrical and Computer Engineering, The University of British Columbia, Vancouver, 2018
* B.Tech. in Electronics and Communication Enginnering, National Institute of Technology Karnataka, India, 2017 

Work experience
======
* ## NSERC Postdoctoral Fellow, University of Waterloo <span style="color:#2C4381">(Oct 2024 - Present)</span> 
  * Consulting on multiple projects in AI for science. 
* ## Researcher, Royal Bank of Canada <span style="color:#2C4381">(July 2024 - Present)</span> 
  * LLM's + evolutionary strategies for spatial optimization using heuristics. 
  * Reinforcement learning from verifiable rewards (RLVR) for learning local natural language heuristics to mimic global graph optimization problems.  
* ## Postdoctoral Fellow, University of Waterloo <span style="color:#2C4381">(Oct 2023 - Present)</span>
  * Developing AI models for decision making regarding boreal afforestation in Canada. 
* ## AI Consultant, Stealth Startups <span style="color:#2C4381">(Jun 2023 - Sep 2023)</span> 
  * Designed an AI framework for metal-organic framework (MOF) discovery for carbon capture and storage applications 
  * Explored mutational landscape of carbonic anhydrase using DL methods 
* ## Machine Learning Engineer, <a href="https://www.gandeeva.com/"><u>Gandeeva Therapeutics</u></a> <span style="color:#2C4381">(Feb 2023 - May 2023)</span> 
  * Integrated molecular dynamics based conformation search with deep learning based protein design for protein affinity maturation. Gandeeva will use this platform for their future protein affinity maturation campaigns.
  * Tested interface scoring tools from CASP15 to drive in-house ppi prediction, which will significantly improve Gandeeva’s target discovery efforts.
  * Tested proof-of-concept target-potential binding partner interface recovery by using protein design to suggest favourable mutations. This will help Gandeeva recover weak but therapeutically relevant interfaces.

* ## Ph.D. Candidate, The University of British Columbia <span style="color:#2C4381">(Jan 2019 - Feb 2023)</span> 
  * Designed representation learning strategies for the <a href="https://kevinbdsouza.github.io/publications/epilstm"><u>epigenome</u></a> and <a href="https://kevinbdsouza.github.io/publications/hiclstm"><u>chromatin structure</u></a>.
  * Tested representations for tasks like pan-celltype element identification, novel element detection, transfer learning to unseen cell types, inference of 3D chromatin structure, and simulating in-silico alterations.

* ## Machine Learning Intern, <a href="https://www.gandeeva.com/"><u>Gandeeva Therapeutics</u></a> <span style="color:#2C4381">(Jun 2022 - Dec 2022)</span> 
  * Designed antibodies using sequence and structure based Machine Learning, antibody folding tools, rosetta energy metrics, and bayesian optimization.

* ## Machine Learning Intern, <a href="https://www.skycope.com/"><u>Skycope Technologies</u></a> <span style="color:#2C4381"> (May 2018 - Sep 2018)</span>
  * Built Skycope's data and machine learning infrastructure.
  * Modified Faster-RCNN, an existing object detection framework, to successfully <a href="https://kevinbdsouza.github.io/publications/frcnn"><u>detect and locate drone signals in the spectrogram</u></a>.
  * Integrated ML into Skycope's existing software stack, which is now it's flagship drone detection software.

* ## M.A.Sc. Research Assistant, The University of British Columbia <span style="color:#2C4381"> (Sep 2017 - Dec 2018)</span>
  * Developed deep learning frameworks for signal detection and hybrid precoding schemes for sequential data.

  
Skills
======
* Machine Learning, Artifical Intelligence, Climate Change, Computational Biology 


Publications and Patents 
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Conferences and Talks
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
* Joint-Secretary at Autism Society of Udupi (ASU), a non-profit organisation in Udupi, India, that aims to create awareness about Autism among parents, teachers, health professionals, students, general public and all the stakeholders so that early diagnosis and early intervention could give the child maximum benefits

* Past mentor at Climate Hub, iGEM, UBC, Lets Talk Science, and Geneskool, Genome BC

Projects
======
  <ul>{% for post in site.projects reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
