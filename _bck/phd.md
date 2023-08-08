---
title: "Representation Learning Strategies for the Epigenome and Chromatin Structure using Recurrent Neural Models"
collection: phd
type: "Thesis"
about: "Thesis"
permalink: /phd/thesis
date: 2023-02-15
---

In this Ph.D. thesis, we propose frameworks for designing informative position-specific representations from epigenomic and structural genomic signals. We use recurrent priors in our analysis owing to the fact that the genome is heavily correlated with nearby positions, and implement them using recurrent neural models. We demonstrate that the representations we learn are helpful for various tasks, including, locating known genomic elements, identifying conserved sites, correlating with established genomic measures, enabling accurate decoding, finding elements that drive 3D conformation, attributing relative positional importance, and performing in-silico modifications. In the process of designing these representations, we study two classes of strategies that differ in their underlying philosophy, namely, autoencoding and categorical encoding. We show that the usefulness of these representations depends on the underlying strategies used while designing them.  

## Epi-LSTM

In the first chapter of my thesis, we employ sequential models to form representations of the epigenome. The availability of thousands of epigenomic assays necessitates compressed representations that summarize the epigenetic landscape of the genome. Many methods have been proposed to form epigenomic representations [1, 2, 3, 4, 5]. Most existing methods produce celltype-specific representations which correspond poorly to conceptualizations of genomic elements. Joint methods aim to improve epigenomic representations by simultaneously using many cell types and sharing position-specific information between them [6, 7, 8, 9]. These methods are accurate but produce different representations for each cell type. On the other hand, stacked methods take data from all available cell types to produce pan-celltype representations, discrete [10, 11] or continuous [12, 13]. Avocado [12] uses separate embeddings for each genomic position, assay type, and cell type and couples this with a feed-forward network that infers unperformed assays. Towards forming pan-celltype epigenomic representations, we propose to improve upon existing feed-forward neural architectures [12] by using a deep long short-term memory (LSTM) [14] recurrent neural network autoencoder called Epi-LSTM [15] to capture the neighborhood dependencies in the epigenomic data. The latent representations from this model capture a variety of genomic phenomena, including frequently interacting regions, promoter-enhancer interactions, gene expression, evolutionary conservation, and replication timing. Moreover, the sequential nature of our model yields smoother epigenomic representations along the genomic axis.

<p align="center">
<img align="center" width=600 height=400 src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/epi-lstm.png?raw=true">
</p>

## Hi-C-LSTM

While epigenomic assays detail the interaction of proteins with the genome, experiments like Hi-C [16, 17] identify genome-wide long-range interactions and enhance our understanding of genome organization by allowing us to analyze the resulting contact matrix using deep learning methods. MATCHA [18] uses hypergraph representation learning to locate multi-way interactions, while Akita [19] uses a convolutional neural network that predicts chromatin contacts from the nucleotide sequence alone and can be used to perform in-silico mutagenesis. Both these methods do not form pairwise Hi-C representations. SNIPER [20] uses a feed-forward network, and SCI [21] uses graph embedding to form Hi-C representations; however, they do not account for the sequential nature of the genome. SNIPER takes the inter-chromosomal contacts as input which, while helpful in locating domain-level structures, might not be the best suited for element-level usage. Moreover, both methods have not demonstrated the capability of performing in-silico mutagenesis. Towards forming position-specific intra-chromosomal representations, in the second chapter of my thesis, we propose to use a deep LSTM model called Hi-C-LSTM [22] that predicts the intra-chromosomal contacts from the pairwise node information.

<p align="center">
<img align="center" width=600 height=400 src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/hi-c-lstm.png?raw=true">
</p>

The representations from our model capture a variety of genomic phenomena and distinguish genomic regions known to cluster in 3D space. Feature importance analysis also reveals association with genomic elements driving 3D conformation and transcription factors such as CCCTC-binding factor (CTCF) and cohesin subunits known to mediate chromatin conformation [23, 24, 25, 26]. A sequential contact generation model is a promising candidate to understand 3D chromatin conformation in light of the results obtained.

<p align="center">
<img align="center" width=500 height=400 src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/map_tasks.png?raw=true">
</p>

To demonstrate that our chromatin conformation model is effective at contact generation, we perform in-silico duplication of a 2.1 Mbp region in chromosome 7 as seen in Cooks syndrome [27] and saw that the predicted Hi-C map shows the presence of a new chromatin domain (neo-TAD). These experiments validate the capability of our model to perform in-silico mutagenesis experiments.

<p align="center">
<img align="center" width=500 height=400 src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/melo_fusion.png?raw=true">
</p>

## Strategies 
<p align="center">
<img align="center" width=600 height=400 src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/strategies.png?raw=true">
</p>

Both the epigenomic model and the chromatin conformation model use an underlying sequential structure to model that genome; however, they differ in the task they are trying to accomplish and strategy they employ. While our epigenomic sequential model reconstructs the epigenomic data given to it as input (autoencoding) [15], our chromatin conformation model seeks to predict the pairwise contact data between nodes by using just the nodes’ positional information (categorical encoding) [22]. As expected, we observe that prediction is much more challenging than reconstruction because of the model’s lack of complete knowledge about the nature of continuous predictions it is trying to make. In the third chapter of my thesis, we aim to study these two strategies further and the crucial differences in their approach to creating genomic representations. These two strategies are fundamentally different, and studying the similarities and differences between the representations they form can provide us insights into the nature of epigenomic and chromatin conformation data.

Our three-part research aims to study the two main strategies of autoencoding and categorical encoding used to form genomic representations assuming a recurrent neural network prior. We demonstrate that while these representations’ utility depends on the prior and the strategy used, they are helpful for non-trivial downstream tasks like pan-celltype classification of genomic phenomena, in-silico mutagenesis, and inference of 3D chromatin structure.


### References 
1. M. Hoffman, O. J. Buske, J. Wang, Z. Weng, J. A. Bilmes, & W. S. Noble. Unsupervised pattern discovery in human chromatin structure through genomic segmentation. Nature methods, 9(5), 473-476. 2012. 
2. J. Ernst, & M. Kellis. ChromHMM: automating chromatin-state discovery and characterization. Nature methods, 9(3), 215. 2012. 
3. N. Day, A. Hemmaplardh, R. E. Thurman, J. A. Stamatoyannopoulos, & W. S. Noble. Unsupervised segmentation of continuous genomic data. Bioinformatics, 23(11), 1424-1426. 2007. 
4. A. Mammana, & HR. Chung. Chromatin segmentation based on a probabilistic model for read counts explains a large portion of the epigenome. Genome Biology 16(1), 151. 2015. 
5. S. G. Coetzee, Z. Ramjan, H. Q. Dinh, B. P. Berman, & D. J. Hazelett. Statehub-statepaintr: rapid and reproducible chromatin state evaluation for custom genome annotation. BioRxiv, 127720. 2017.
6. B. Chen, N. S. Kenari, & M. W. Libbrecht. Continuous chromatin state feature annotation of the human epigenome. bioRxiv, 473017. 2018.
7. Y. Zhang, L. An, F. Yue, & R. C. Hardison. Jointly characterizing epigenetic dynamics across multiple human cell types. Nucleic acids research 44(14), 6721-6731. 2016. 
8. M. W. Libbrecht, M. M. Hoffman, J. A. Bilmes, & W. S. Noble. Entropic graph-based posterior regularization: Extended version. In Proceedings of the International Conference on Machine Learning. 2015. 
9. M. W. Libbrecht, F. Ay, M. M. Hoffman, D. M. Gilbert, J. A. Bilmes, & W. S. Noble. Joint annotation of chromatin state and chromatin conformation reveals relationships among domain types and identifies domains of cell-type-specific expression. Genome Research, 25(4), 544-557. 2015.
10. M. M. Hoffman, J. Ernst, S. P. Wilder, A. Kundaje, R. S. Harris, M. Libbrecht, et al. Integrative annotation of chromatin elements from ENCODE data. Nucleic acids research, 41(2), 827-841. 2013. 
11. M. W. Libbrecht, O. L. Rodriguez, Z. Weng, J. A. Bilmes, M. M. Hoffman, & W. S. Noble. A unified encyclopedia of human functional DNA elements through fully automated annotation of 164 human cell types. Genome Biology, 20(1), 180. 2019. 
12. J. Schreiber, T. Durham, J. Bilmes, & W. S. Noble. Multi-scale deep tensor factorization learns a latent representation of the human epigenome. BioRxiv, 364976. 2019.
13. T. J. Durham, M. W. Libbrecht, J. J. Howbert, J. Bilmes, & W. S. Noble. PREDICTD parallel epigenomics data imputation with cloudbased tensor decomposition. Nature communications, 9(1), 1402. 2018.
14. S. Hochreiter, & J. Schmidhuber. Long short-term memory. Neural computation, 9(8), 1735-1780. 1997.
15. K. B. Dsouza, A. Y. Li, V. Bhargava, & M. W. Libbrecht. Latent representation of the human pan-celltype epigenome through a deep recurrent neural network. IEEE/ACM Transactions on Computational Biology and Bioinformatics. 2021. 
16. N. L Van Berkum, E. Lieberman-Aiden, L. Williams, M. Imakaev, A. Gnirke, L. A. Mirny, et al. Hi-C: a method to study the three-dimensional architecture of genomes. JoVE (Journal of Visualized Experiments). 39:e1869. 2010.
17. S. S. Rao, M. H. Huntley, N. C. Durand, E. K. Stamenova, I. D. Bochkov, J. T. Robinson, et al. A 3D map of the human genome at kilobase resolution reveals principles of chromatin looping. Cell. 159:1665-80. 2014.
18. R. Zhang, & J. Ma. Probing multi-way chromatin interaction with hypergraph representation learning. Cell Systems. 10:397-407. 2020.
19. G. Fudenberg, D. R. Kelley, & K. S. Pollard. Predicting 3D genome folding from DNA sequence with Akita. Nature Methods. 17:1111-1117. 2020.
20. K. Xiong, & J. Ma. Revealing Hi-C subcompartments by imputing inter-chromosomal chromatin interactions. Nature communications. 10. 2019.
21. H. Ashoor, X. Chen, W. Rosikiewicz, J. Wang, A. Cheng, P. Wang, et al. Graph embedding and unsupervised learning predict genomic sub-compartments from HiC chromatin interaction data. Nature communications. 11:1. 2020.
22. K. B. Dsouza, A. Maslova, A. J. Ediem, M. Merkenschlager, V. K. Bhargava, & M. W. Libbrecht. Hi-C-LSTM: Learning representations of chromatin contacts using a recurrent neural network identifies genomic drivers of conformation. To be Submitted. Genome Biology. 
23. S. Cuddapah, R. Jothi, D. E. Schones, T. Y. Roh, K. Cui, & K. Zhao. Global analysis of the insulator binding protein CTCF in chromatin barrier regions reveals demarcation of active and repressive domains. Genome research. 19:24-32. 2009.
24. J. E. Phillips, & V. G. Corces. CTCF: master weaver of the genome. Cell. 137:1194-211. 2009.
25. E. Splinter, H. Heath, J. Kooren, R. J. Palstra, P. Klous, F. Grosveld, et al. CTCF mediates long-range chromatin looping and local histone modification in the $\beta$-globin locus. Genes & development. 20:2349-54. 2006.
26. A. L. Sanborn, S. S. Rao, S. C. Huang, N. C. Durand, M. H. Huntley, A. I. Jewett, et al. Chromatin extrusion explains key features of loop and domain formation in wild-type and engineered genomes. Proceedings of the National Academy of Sciences. 112:E6456-65. 2015.
27. U. S. Melo, R. Schöpflin, R. Acuna-Hidalgo, M. A. Mensah, B. Fischer-Zirnsak, M. Holtgrewe, et al. Hi-C identifies complex genomic rearrangements and TAD-shuffling in developmental diseases. The American Journal of Human Genetics. 106(6):872-884. 2020.



