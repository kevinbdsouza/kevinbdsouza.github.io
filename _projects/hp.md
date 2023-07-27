---
title: "Hybrid Precoding for mmWave Massive MIMO OFDM"
collection: projects
about: "Partially Connected Structure"
type: "Hybrid Precoding"
permalink: /projects/hp
date: 2018-03-31
---

Hybrid precoding, a combination of digital and analog precoding, is an alternative to traditional precoding methods in massive MIMO systems with a large number of antenna elements and has shown promising results recently. In this paper, we implement a parallel framework to make hybrid precoding competitive in fast-fading environments. A low-complexity algorithm which exploits the block diagonal phase-only nature of the analog precoder in a partially connected structure is proposed to arrive at a hybrid precoding solution for a multi-carrier single-user system using orthogonal frequency division multiplexing (OFDM). The original problem is broken down into two subproblems of finding the magnitude and the phase components which are solved independently. A per-RF chain power constraint is introduced instead of the sum power constraint over all antennas which are much more practical in real systems. An alternating version of the same algorithm is proposed for increased spectral-efficiency gains. Complexity and run-time analysis demonstrate the advantage of the proposed algorithm over existing hybrid precoding schemes for partially connected structure in an OFDM setting. The simulation results reveal certain insights about the partially connected structure and the tradeoffs that have to be made to make it workable in a real wideband system.

<p align="center">
<img align="center" width=600 height=400 src="https://github.com/kevinbdsouza/kevinbdsouza.github.io/blob/master/files/hp.png?raw=true">
</p>

Read the <a href="https://ieeexplore.ieee.org/document/8490877"><u>Paper</u></a>.
