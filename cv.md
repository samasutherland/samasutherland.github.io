---
layout: page
title: Curriculum Vitae
permalink: /cv/
---

*Quantum computing PhD with experience in numerical methods and applied machine learning. I specialise in differentiable programming and tensor networks, with keen interest in applying these techniques in frontier AI models.*

[Download abridged PDF version of this CV]({{ '/assets/Sam-Sutherland-CV-06-09-2026.pdf' | relative_url }})

---

## Experience

**Silicon Quantum Computing — PhD student (2021–2026)**

University of New South Wales, Sydney

- Built multi-node DDP training with gradient-propagating collectives in PyTorch for Hamiltonian learning
- Derived and implemented a solver-agnostic backward pass for tensor network eigensolvers, eliminating the dependence of gradient cost and memory on solver iteration count
- Designed and tested a novel reservoir computing device, “Watermelon”, based on an array of phosphorus-doped silicon quantum dots, which became a primary revenue stream for the company

**Optiver — Quantitative researcher intern (November 2023 – February 2024)**

Sydney

- Completed accelerated training in market making and quantitative trading
- Evaluated a news-based trading signal on Bloomberg headlines; found no edge at actionable latency
- Developed improved estimators of intermediate-scale volatility in global equity indices

**Ocado Technology — Innovation engineer (April – December 2020)**

Hatfield

- Applied self-supervised pretraining (MoCo, BYOL) to warehouse basket imagery for automated pick verification; found limited gains over supervised training given the abundance of implicit labels
- Evaluated multiphysics simulation packages and advised on their usage

**Ocado Technology — Machine learning internship (July – December 2019)**

Hatfield

- Built an autonomous navigation stack on a mobile robot platform: SLAM-based mapping and routing with a vision model for obstacle detection, demonstrated end-to-end in a live office environment

**University of Oxford — MPhys research project (2018–2019)**

Oxford

- Trained a model on satellite data to detect Pockets of Open Cells in clouds on an unprecedented scale
- Paper resulting from subsequent analysis won the climate change AI workshop best paper award at ICML 2019

**Centre for Applied Superconductivity — Summer research project**

12 weeks in 2018, University of Oxford — simulation using COMSOL

- Published an information page and video on the website detailing my work to the general public
- Worked autonomously when my supervisor left for a month to perform research abroad: completed the tasks she had set and used the results of those to guide the research
- Worked closely with the team to develop simulations, meeting with them and presenting my work every fortnight

**Perm State University — Computational fluid dynamics internship**

6 weeks in 2017, Russia

- Communicated effectively with a multi-cultural group of students and professors to maximise the value of the course on simulation techniques
- Implemented and optimised these techniques in FORTRAN to achieve results comparable to papers in this field
- Utilised their linux compute cluster to run simulations that I had parallelised myself using MPI

---

## Independent Research

**[Second-Order Jacobian Lens]({{ '/projects/second-order-jacobian-lens/' | relative_url }}) — Interpretability study (2026)**

- Showed low-rank sketching of the Jacobian Lens with task-agnostic bases recovers no more than a random subspace, while a lens-derived basis reaches full performance at 10% of the dimensions
- Showed the second-order interaction of token-gated routing is step-like, so gradient-based estimation reports false negatives for most token pairs

**[little Language Models]({{ '/llms/' | relative_url }}) — Transformer architecture experiments (2026)**

- Built a config-driven, Pydantic-validated framework that compares transformer variants at equal compute
- Made every result reproducible from a commit: containerised build, ephemeral GPU provisioning, and training through evaluation run automatically on push

---

## Education

**UNSW — Doctor of Physics (2021–2026)**

*Thesis submitted, under examination*

- Primary supervisor: Prof. Michelle Simmons AC FRS

**University of Oxford — Master of Physics (2015–2019)**

*1st class*

- Studied Quantum Information Processing, Theoretical Physics, and Lasers and Optics in my final year
- Academic scholarships awarded for second and third year results

**Wanstead High School — A levels and GCSEs**

*2013–2015:* Maths A\*, Further Maths A\*, Physics A\*, Chemistry A, Music A  

*GCSEs 2008–2013:* 8 A\*’s and 4 A’s including A\* in Maths, Physics and English Literature

---

## Skills

**Numerical**

- Tensor network methods  
- Vectorised methods  
- Quantum simulation  

**ML & AI**

- Transformer architectures  
- Differentiable programming  
- Optimisation methods  
- Self-supervised learning  
- Distributed training  

**Implementation**

- Python  
- PyTorch  
- Custom autograd  

---

## Publications

- **Nature** — M. B. Donnelly, Y. Chung, R. Garreis, …, **S. Sutherland** *et al.*, [“Large-scale analogue quantum simulation using atom dot arrays,”](https://doi.org/10.1038/s41586-025-10053-7) (2026).

- **In preparation** — **S. A. Sutherland**, M. B. Donnelly, J. G. Keizer *et al.*, “A precision engineered Fermionic lattice for quantum-enhanced machine learning,” (2026).

- **ACS Nano** — A. D. Tranter, L. Kranz, **S. Sutherland** *et al.*, [“Machine Learning-Assisted Precision Manufacturing of Atom Qubits in Silicon,”](https://doi.org/10.1021/acsnano.4c00080) vol. 18, no. 30 (2024).

- **PRB** — M. Ghini, M. Bristow, J. C. A. Prentice, **S. Sutherland** *et al.*, [“Strain tuning of nematicity and superconductivity in single crystals of FeSe,”](https://doi.org/10.1103/PhysRevB.103.205139) (2021).

- **GRL** — D. Watson-Parris, **S. A. Sutherland**, M. W. Christensen *et al.*, [“A Large-Scale Analysis of Pockets of Open Cells and Their Radiative Impact,”](https://doi.org/10.1029/2020GL092213) (2021).

---

## Patents

- **WO/2024** — **S. Sutherland**, S. K. Gorman, C. Myers, and M. Y. Simmons, [“Quantum machine learning devices and methods,”](https://patents.google.com/patent/WO2024007054A1/en) *WO2024007054A1* (2024).

- **WO/2024** — M. Y. Simmons, S. K. Gorman, L. Kranz, **S. Sutherland** *et al.*, [“Advanced quantum processing systems,”](https://patents.google.com/patent/WO2024073818A1/en) *WO2024073818A1* (2024).

- **WO/2023** — S. K. Gorman, M. Y. Simmons, J. Keizer, …, **S. Sutherland**, [“Methods and systems for analogue quantum computing,”](https://patents.google.com/patent/WO2023064999A1/en) *WO2023064999A1* (2023).

- **AU/2023** — Silicon Quantum Computing Pty Ltd, [“Methods for fabricating advanced quantum processing systems,”](https://patents.google.com/patent/AU2022903898A0/en) *AU2022903898A0* (2023).
