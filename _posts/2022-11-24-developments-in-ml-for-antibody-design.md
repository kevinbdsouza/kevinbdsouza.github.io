---
title: 'Developments in machine learning for antibody design'
date: 2022-11-24
permalink: /posts/2022/11/developments-in-ml-for-antibody-design
tags:
  - antibody design 
  - artificial intelligence 
  - biologics 
---

Protein structure and sequence modeling has seen a fresh wave of resurgence in the last couple of years owing to some interesting developments in machine learning (ML) and deep learning (DL) based techniques. These techniques appear in a variety of flavours including using Equivariant neural network modules to respect the structural properties of 3D macromolecules, deeper networks that can benefit from the increased available experimental structures, powerful node-to-node relationship learners like transformers, and masked language modeling on the protein sequence space to learn evolutionary information. While structure prediction methods like AlphaFold (AF) [1] and RosettaFold (RF) [2] have become ubiquitious in computational structural biology, there remain challenges to be tackled on multiple fronts, where ML will play an important role. 

From here on I intend to use ML as a singular term for the group of techniques that learn parameters of models from data using gradient descent, irrespective of the architecture or the size of the models employed. 

Proteins come in many shapes and forms and perform a multitude of functions in the body, and antibodies are one such class of proteins. Antibodies are an important group of proteins that are key players in the workings of immune system and require B cells and T cells for their production. Foreign protein antigens that enter the body activate the B-cells in a T cell-dependent manner. Each B cell has multitudes of identical B cell receptors (BCRs) expressed on the surface that selectively bind to a particular epitope of the antigen. The protein antigen that binds to a BCR is taken up into the B cell through receptor-mediated endocytosis [3], degraded, and presented to Helper T cells as peptide fragments in complex with MHC-II molecules on the cell membrane. Helper T cells recognize and bind these MHC-II-peptide complexes through their T cell receptors (TCR). After TCR-MHC-II-peptide binding, T cells express certain surface proteins that are necessary for B cell activation. Activated B cells result in both short-lived variants that produce weak antibodies and long-lived variants that produce high affinity antibodies by a process called somatic hypermutation (SHM). SHM is a process in which point mutations accumulate in the variable domains of the antibody, particularly in the hypervariable regions, of both the heavy and light chains, at rates million-fold higher than the background mutation rates observed in other genes. While SHM does not distinguish between favorable and unfavorable mutations, the selection process selects for B cells that produce the highest-affinity antibodies. The ability to do this computationally, that is given any protein target, designing antibodies that can bind to it with high affinity (without multiple rounds of extensive mutations and selection), is of extremely high therapeutic relevance. Furthermore, the use of antibodies as therapeutics is relatively under-explored with less than 105 FDA approved antibodies to date.  

While general protein structure prediction from sequence has reached low RMSD with respect to experimental structures, antibody structure prediction is more challenging due to the presence of highly diverse and flexible loops called complimentarity determining regions (CDRs). Antibodies are Y-shaped molecules with a fixed region called fragment crystallizable (Fc) region and two arms which are called fragment antibody binding (Fab) regions. A variable number of intrachain disulphide bonds hold the Fc and Fab together, where the number varies depending on the type of antibody. The Fabs are comprised of a constant domain and a variable domain, where the constant domain is relatively conserved across different antibodies and the variable domain shows higher variablity. Most of the variability in the variable domain comes from the CDRs at the tips of these domains that form a dominant part of the paratope that binds to the antigen epitope. Each arm is also segregated into two separate chains, the heavy chain and the light chain, with CDR-H1, H2, H3 in the heavy chain and CDR-L1, L2, L3 in the light chain. Among the 6 CDR regions, CDR-H3 plays the most important role in binding, is the longest and also the most diverse.   
Given the peculiarities of antibodies, designing them computationally to bind to a target protein has been challenging. I partition this broader problem into three categories that differ in their available inputs and end goal. 

* In-silico affinity maturation
* CDR infilling
* De novo design    

In-silico affinity maturation aims to enhance the binding affinity of antibodies that are known to bind to the target. This is primarily achieved by a framework similar to the one shown in Fig. 1. Representations of BCR repertoires are first learned using either antibody language models or other representation learning techniques. Antibody language models are increasingly becoming powerful at creating useful representations which can then be used for downstream tasks (examples include IgLM [4], AbLang [5]). These representations are then used to train a regression model that predicts the target binding affinity of sequences by using measured experimental data as the ground truth. The trained regression model is then used in an in-silico directed evolution campaign to generate new sequences. The in-silico directed evolution process can take many forms including active learning or bayesian optimization. RESP AI [6], the geometric deep learning (GDL) model by Peng [7], and the language models assisted bayesian optimization framework by Walsh [8] come under this first category.        

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/aff_mat.png?raw=true" width="400"/>
</p>
<p align="center">
<em> <font size="2"> Figure 1. Framework for in-silico affinity maturation.</font> </em>
</p>

The second category pertains to designing alternate CDR regions that bind to an antigen with greater affinity given an existing antibody-antigen complex structure. This is achieved by a framework similar to the one shown in Fig. 2. Existing antibody-antigen complexes are used as training data to train a model that can predict either the sequence, or the backbone structure, or both simultaneously. The methods in this category include MEAN [9], the model by Absci [10], AntiDesigner [11], DiffAb [12], Fragment-based method by Sormanni [13], C-RefineGNN [14], and AbODE [15]. Some of these methods use diffusion models, some work with partial differential equations (PDEs), and most methods work on top of graphs. CDR evolution in the sequence space is also another class of methods in this category, shown in Fig. 3. These methods use antibody language model log probabilities to chose fitter CDR regions in the sequence space. Examples inlcude methods like IgLM [4], AbMAP [16], AbLang [5], ESM based method by Kim [17], and ReprogBert [18]. 
 
<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/cdr_infill.png?raw=true" width="300"/>
</p>
<p align="center">
<em> <font size="2"> Figure 2. CDR sequence/structure infilling scenario. </font> </em>
</p>

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/cdr_infill_2.png?raw=true" width="300"/>
</p>
<p align="center">
<em> <font size="2"> Figure 3. CDR sequence evolution scenario. </font> </em>
</p>

The third category consists of designing the entire antibody from scratch such that it can bind to a given target. While the first category is epitope agnostic, the second and the third categories are epitope specific, that is, the antibody needs to be designed such that it binds to a particular epitope of interest on the antigen. The problem of designing the antibody can be broken down into two subproblems of designing the framework region (Fab minus the hypervariable CDR regions) and designing the CDR regions (The Fc region is same for a given class of antibody in the species and both arms of the antibody can be identical). 

The framework of the antibody can be designed one of three ways. First, use existing framework strcutures from databases like SAbDab [19]. Second, use antibody structure prediction methods to predict the structures of the framework regions in the antibody sequence databases like the observed antibody space (OAS) [20], and third, de novo design the whole framework structure. There are methods that go the third route and try to either diffuse the framework stucture like Ig-VAE [21] and PF-ABGEN [22] (Fig. 4), or design it using cross-Beta motif rules such as the method by Marcos [23]. Designing a de novo framework is often not necessary because of the vast available sequence repertoires and good antibody structure prediction results for the framework region. Therefore, using the OAS in conjunction with antibody structure prediction methods is a good start given the small number of eperimentally available framework structures.    

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/diffuse.png?raw=true" width="200"/>
</p>
<p align="center">
<em> <font size="2"> Figure 4. De novo diffusion of antibody frameworks. </font> </em>
</p>

Important components of the epitope-specific de novo computational antibody design pipeline are shown in Fig. 5. Designing a reasonable scoring function that can access the quality of the antibody-antigen interface is the first step in this pipeline. Wallner [24] has shown in CASP15 that sampling multiple structural models and then using an interface scoring function to filter these results in siginificant improvements in protein complex structure prediction. Models that perform estimation of model accuracy (EMA) and access interface quality are good candidates for such a scoring function, for example, VoroIF-GNN [25], MULTICOM_qa [26], and ModFOLDdock [27]. The docking potential from rigid body docking methods like ZDOCK [28], HDOCK [29], and HADDOCK3 [30] can be used to generate thousands of antibody-antigen interfaces for which experimental ground truth exists. The generated poses can then be re-scored using a combination of re-scoring methods like DLAB-Re [31], AF2 composite score [32], Graphinity [33], solvated interaction energy (SIE) [34], VoroIF-GNN [25], Rosetta energy, and hydrogen bond score. In fact, training a surrogate ML model to learn a composite score directly is an option. This surrogate model would combine the sequence features (e.g: representations from antibody LMs), with the landscape of re-scoring potentials to predict a normalized composite score for each antibody candidate sequence.  

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/pipeline.png?raw=true" width="700"/>
</p>
<p align="center">
<em> <font size="2"> Figure 5. Antibody design pipeline. </font> </em>
</p>

Once a reasonable scoring function is designed, the next is to perform framework selection and virtual screening. In order to do this, we first filter redundant, non-human framework sequences and sequences with $$size(indels) > K$$ using ANARCI status [35]. We can also use $$Redundancy > N$$ to retain sequences as an additional criterion for robustness. Next, we cluster concatenated CDRs on sequences identity into $$A$$ clusters, and further cluster each cluster on the framework minus the CDR-H3 region using alignment and clustering tools like MMseqs2 [36]. Once the clustering step is completed, we pick a representative from each CDR cluster and generate structures using antibody folding tools like ABodyBuilder2 [37]. This is followed by generating docked poses (restricted near the epitope) and scoring using the designed composite scoring function. Finally, we repeat this process for inner framework clusters for the $$B$$ best CDR clusters, and select the top $$C$$ best scoring frameworks. 

Once the best frameworks have been chosen, The CDRs can be diversified using methods that co-design the structure and sequence given the antibody-antigen pose, like MEAN [9], and DiffAb [12]. We can potentially focus the diversification on the most important CDR for binding which is CDR-H3 as shown in Fig. 6. The diversified CDR-H3 poses are then scored using the surrogate model that predicts the composite score. The common issue with desiginig such a scoring function is the scarcity of experimental antibody-antigen complexes. Synthetic data from simulation frameworks such as Absolut!? [38] may help, however, experimental validation of such frameworks is still lacking. 

<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/cdr_h3.png?raw=true" width="700"/>
</p>
<p align="center">
<em> <font size="2"> Figure 6. CDR-H3 backbone and sequence diversification and interface scoring. </font> </em>
</p>

One solution for data scarcity, is a batched bayesian optimization (BO) or active learning (AL) framework that continualy refines the scoring function after batches of wet lab testing, before applying the final post-design filters. Implementing BO/AL along with the re-scoring potentials would require combining the sequence-specific binding affinity ($$K_D$$) values from wet lab experiments and features from antibody LMs with the landscape of pose-specific re-scoring potentials. The predicted composite score would be a normalized version of $$K_D$$ values. Note that the positive hits from the wet lab don't give us information about the epitope of binding, which can only be obtained through imaging. Therefore, there is a possibility that some of the resulting sequences from this approach bind the incorrect epitope.   

The final step consists of dropping sequences that have poor developability and biophysical properties (Fig. 7). This includes using methods like AggScore [39] to check for aggregation prone regions, CamSol [40] to find insoluble patches, DeepSCM [41] to remove high viscosity candidates, BioPhi [42] to evaluate humanness, and TransMHCII [43]/NetMHCIIpan 4 [44] to check for immunogenicity. Therapeutic Antibody Profiler [45] is another important tool that uses surface hydrophobicity and charge near the CDR regions to check how similar the candidate antibody is to the therapeutically approved antibodies. It is possible that sequences that have a high predicted composite binding score might have to be discarded because of some of these filters. Rescue strategies may still help to retain some of those sequences like using Hu-mAb [46] or BioPhi [42] to humanize failed humanness candidates, or use a Multi-objective optimization strategy as described by Sormanni [47] with respect to the failed properties.  


<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/filters.png?raw=true" width="500"/>
</p>
<p align="center">
<em> <font size="2"> Figure 7. Post-design developability filters. </font> </em>
</p>

Machine learning will play an important role in addressing some of the extant challenges in computational antibody design. In most cases, the answer to the question "which is the best/most promising approach?", depends on the available wet lab infrastructure. Designing good interface scoring functions for antibody-antigen interfaces is definitely one of the central challenges. In situations where wet lab feedback is available, refining scoring functions using BO/AL feedback loop is key. Is it possible to design a general purpose scoring function that works reasonably well with all targets? Given the current available experimental antibody-antigen complex structure data, the answer is probably not. Presently, reliable antibody folding plays a major role in modeling, however, improvements in the modeling of CDR-H3 conformational diversity will be crucial (like the recent MLSA [48]). The canonical CDRs can also be factored in during design (using canonical clustering frameworks like PyIgClassify2 [49]), but the question remains as to how large a design space can we actually explore and whether it is better to focus more on the CDR-H3. While the joint sequence-structure CDR-H3 co-design methods such as MEAN [9] and DiffAb [12] are useful for smart sampling, they are still immature and need to improve further. Antibody LMs will also have a siginificant impact on navigating the sequence space efficiently as demonstrated by the recent xTrimoPGLM [50]. With all that said, it is not long before the data becomes available and our models become vastly more powerful than they are currently.   


References:

1. Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., ... & Hassabis, D. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873), 583-589.
2. Baek, M., Anishchenko, I., Humphreys, I., Cong, Q., Baker, D., & DiMaio, F. (2023). Efficient and accurate prediction of protein structure using RoseTTAFold2. bioRxiv, 2023-05.
3. Gao, H., Shi, W., & Freund, L. B. (2005). Mechanics of receptor-mediated endocytosis. Proceedings of the National Academy of Sciences, 102(27), 9469-9474.
4. Shuai, R. W., Ruffolo, J. A., & Gray, J. J. (2021). Generative language modeling for antibody design. bioRxiv, 2021-12.
5. Olsen, T. H., Moal, I. H., & Deane, C. M. (2022). AbLang: an antibody language model for completing antibody sequences. Bioinformatics Advances, 2(1), vbac046.
6. Parkinson, J., Hard, R., & Wang, W. (2023). The RESP AI model accelerates the identification of tight-binding antibodies. Nature Communications, 14(1), 454.
7. Shan, S., Luo, S., Yang, Z., Hong, J., Su, Y., Ding, F., ... & Peng, J. (2022). Deep learning guided optimization of human antibody against SARS-CoV-2 variants with broad neutralization. Proceedings of the National Academy of Sciences, 119(11), e2122954119.
8. Li, L., Gupta, E., Spaeth, J., Shing, L., Jaimes, R., Engelhart, E., ... & Walsh, M. E. (2023). Machine learning optimization of candidate antibody yields highly diverse sub-nanomolar affinity antibody libraries. Nature Communications, 14(1), 3454.
9. Kong, X., Huang, W., & Liu, Y. (2022). Conditional antibody design as 3d equivariant graph translation. arXiv preprint arXiv:2208.06073.
10. Shanehsazzadeh, A., Bachas, S., McPartlon, M., Kasun, G., Sutton, J. M., Steiger, A. K., ... & Meier, J. (2023). Unlocking de novo antibody design with generative artificial intelligence. bioRxiv, 2023-01.
11. Tan, C., Gao, Z., & Li, S. Z. (2023). Cross-Gate MLP with Protein Complex Invariant Embedding is A One-Shot Antibody Designer. arXiv e-prints, arXiv-2305.
12. Luo, S., Su, Y., Peng, X., Wang, S., Peng, J., & Ma, J. (2022). Antigen-specific antibody design and optimization with diffusion-based generative models for protein structures. Advances in Neural Information Processing Systems, 35, 9754-9767.
13. Aguilar Rangel, M., Bedwell, A., Costanzi, E., Taylor, R. J., Russo, R., Bernardes, G. J., ... & Sormanni, P. (2022). Fragment-based computational design of antibodies targeting structured epitopes. Science Advances, 8(45), eabp9540.
14. Jin, W., Wohlwend, J., Barzilay, R., & Jaakkola, T. (2021). Iterative refinement graph neural network for antibody sequence-structure co-design. arXiv preprint arXiv:2110.04624.
15. Verma, Y., Heinonen, M., & Garg, V. (2023). AbODE: Ab Initio Antibody Design using Conjoined ODEs. arXiv preprint arXiv:2306.01005.
16. Singh, R., Im, C., Sorenson, T., Qiu, Y., Wendt, M., Nanfack, Y., ... & Berger, B. (2023). Learning the Language of Antibody Hypervariability. bioRxiv, 2023-04.
17. Hie, B. L., Shanker, V. R., Xu, D., Bruun, T. U., Weidenbacher, P. A., Tang, S., ... & Kim, P. S. (2023). Efficient evolution of human antibodies from general protein language models. Nature Biotechnology.
18. Melnyk, I., Chenthamarakshan, V., Chen, P. Y., Das, P., Dhurandhar, A., Padhi, I., & Das, D. (2023). Reprogramming Pretrained Language Models for Antibody Sequence Infilling.
19. Dunbar, J., Krawczyk, K., Leem, J., Baker, T., Fuchs, A., Georges, G., ... & Deane, C. M. (2014). SAbDab: the structural antibody database. Nucleic acids research, 42(D1), D1140-D1146.
20. Olsen, T. H., Boyles, F., & Deane, C. M. (2022). Observed Antibody Space: A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences. Protein Science, 31(1), 141-146.
21. Eguchi, R. R., Choe, C. A., & Huang, P. S. (2022). Ig-VAE: Generative modeling of protein structure by direct 3D coordinate generation. PLoS computational biology, 18(6), e1010271.
22. Huang, C., Liu, Z., Bai, S., Zhang, L., Xu, C., WANG, Z., ... & Xiong, Y. (2023, May). PF-ABGen: A Reliable and Efficient Antibody Generator via Poisson Flow. In ICLR 2023-Machine Learning for Drug Discovery workshop.
23. Chidyausiku, T. M., Mendes, S. R., Klima, J. C., Nadal, M., Eckhard, U., Roel-Touris, J., ... & Marcos, E. (2022). De novo design of immunoglobulin-like domains. Nature Communications, 13(1), 5661.
24. Wallner, B. (2022). AFsample: Improving Multimer Prediction with AlphaFold using Aggressive Sampling. bioRxiv, 2022-12.
25. Olechnovic, K., & Venclovas, C. (2023). VoroIF-GNN: Voronoi tessellation-derived protein-protein interface assessment using a graph neural network. bioRxiv, 2023-04.
26. Roy, R. S., Liu, J., Giri, N., Guo, Z., & Cheng, J. (2023). Combining pairwise structural similarity and deep learning interface contact prediction to estimate protein complex model accuracy in CASP15. Proteins: Structure, Function, and Bioinformatics.
27. Edmunds, N. S., Alharbi, S. M., Genc, A. G., Adiyaman, R., & McGuffin, L. J. (2023). Estimation of model accuracy in CASP15 using the M odFOLDdock server. Proteins: Structure, Function, and Bioinformatics.
28. Pierce, B. G., Wiehe, K., Hwang, H., Kim, B. H., Vreven, T., & Weng, Z. (2014). ZDOCK server: interactive docking prediction of protein–protein complexes and symmetric multimers. Bioinformatics, 30(12), 1771-1773.
29. Yan, Y., Tao, H., He, J., & Huang, S. Y. (2020). The HDOCK server for integrated protein–protein docking. Nature protocols, 15(5), 1829-1852.
30. Dominguez, C., Boelens, R., & Bonvin, A. M. (2003). HADDOCK: a protein− protein docking approach based on biochemical or biophysical information. Journal of the American Chemical Society, 125(7), 1731-1737.
31. Schneider, C., Buchanan, A., Taddese, B., & Deane, C. M. (2022). DLAB: deep learning methods for structure-based virtual screening of antibodies. Bioinformatics, 38(2), 377-383.
32. Gaudreault, F., Corbeil, C. R., & Sulea, T. (2022). Enhanced antibody-antigen structure prediction from molecular docking using AlphaFold2. bioRxiv, 2022-12.
33. Hummer, A. M., Schneider, C., Chinery, L., & Deane, C. M. (2023). Investigating the Volume and Diversity of Data Needed for Generalizable Antibody-Antigen ΔΔG Prediction. bioRxiv, 2023-05.
34. Purisima, E. O., Corbeil, C. R., Gaudreault, F., Wei, W., Deprez, C., & Sulea, T. (2023). Solvated interaction energy: from small-molecule to antibody drug design. Frontiers in Molecular Biosciences, 10, 1210576.
35. Dunbar, J., & Deane, C. M. (2016). ANARCI: antigen receptor numbering and receptor classification. Bioinformatics, 32(2), 298-300.
36. Steinegger, M., & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature biotechnology, 35(11), 1026-1028.
37. Abanades, B., Wong, W. K., Boyles, F., Georges, G., Bujotzek, A., & Deane, C. M. (2023). ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins. Communications Biology, 6(1), 575.
38. Robert, P. A., Akbar, R., Frank, R., Pavlović, M., Widrich, M., Snapkov, I., ... & Greiff, V. (2022). Unconstrained generation of synthetic antibody–antigen structures to guide machine learning methodology for antibody specificity prediction. Nature Computational Science, 2(12), 845-865.
39. Sankar, K., Krystek Jr, S. R., Carl, S. M., Day, T., & Maier, J. K. (2018). AggScore: prediction of aggregation‐prone regions in proteins based on the distribution of surface patches. Proteins: Structure, Function, and Bioinformatics, 86(11), 1147-1156.
40. Oeller, M., Kang, R., Bell, R., Ausserwöger, H., Sormanni, P., & Vendruscolo, M. (2023). Sequence-based prediction of pH-dependent protein solubility using CamSol. Briefings in Bioinformatics, 24(2), bbad004.
41. Lai, P. K. (2022). DeepSCM: An efficient convolutional neural network surrogate model for the screening of therapeutic antibody viscosity. Computational and Structural Biotechnology Journal, 20, 2143-2152.
42. Prihoda, D., Maamary, J., Waight, A., Juan, V., Fayadat-Dilman, L., Svozil, D., & Bitton, D. A. (2022, December). BioPhi: A platform for antibody design, humanization, and humanness evaluation based on natural antibody repertoires and deep learning. In MAbs (Vol. 14, No. 1, p. 2020203). Taylor & Francis.
43. Yu, X., Negron, C., Huang, L., & Veldman, G. (2023). TransMHCII: a novel MHC-II binding prediction model built using a protein language model and an image classifier. Antibody Therapeutics, 6(2), 137-146.
44. Reynisson, B., Alvarez, B., Paul, S., Peters, B., & Nielsen, M. (2020). NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by concurrent motif deconvolution and integration of MS MHC eluted ligand data. Nucleic acids research, 48(W1), W449-W454.
45. Raybould, M. I., & Deane, C. M. (2022). The therapeutic antibody profiler for computational developability assessment. Therapeutic Antibodies: Methods and Protocols, 115-125.
46. Marks, C., Hummer, A. M., Chin, M., & Deane, C. M. (2021). Humanization of antibodies using a machine learning approach on large-scale repertoire data. Bioinformatics, 37(22), 4041-4047.
47. Rosace, A., Bennett, A., Oeller, M., Mortensen, M. M., Sakhnini, L., Lorenzen, N., ... & Sormanni, P. (2023). Automated optimisation of solubility and conformational stability of antibodies and proteins. Nature Communications, 14(1), 1937.
48. Giovanoudi, E., & Rafailidis, D. (2023). Multi-Task Learning with Loop Specific Attention for CDR Structure Prediction. arXiv preprint arXiv:2306.13045.
49. Kelow, S., Faezov, B., Xu, Q., Parker, M. I., Adolf-Bryfogle, J., & Dunbrack Jr, R. L. (2022). A penultimate classification of canonical antibody CDR conformations. bioRxiv, 2022-10.
50. Chen, B., Cheng, X., Geng, Y. A., Li, S., Zeng, X., Wang, B., ... & Song, L. (2023). xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein. bioRxiv, 2023-07. 





