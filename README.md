# RIVIEW: Research Integrity - Virtual Image Examination Workspace
*Former SILA: A System for Scientific Image Analysis*

Source codes to reproduce the experimental results of our Nature Scientific Reports paper titled "SILA: a system for scientifc image
analysis", available at https://www.nature.com/articles/s41598-022-21535-3.
![SILA workflow.](system-workflow.png)

## Scientific Papers Dataset
To support assessing the efficacy of the system proposed in the paper, we collected a dataset of scientific manuscripts containing annotated image manipulations and inadvertent figure reuse.
This dataset is available in the following project branch:  
* [RIVIEW Dataset](https://git.io/JcZsX)

*Disclaimer.*
Each paper in the dataset is referred to by its public Digital Object Identifier (DOI).
Inclusion in the dataset does not necessarily signify that an article is the subject of any research misconduct practice, allegation, or proceeding.

## System Modules
The implementation and test scripts of the modules composing the proposed system are individually made available in the following project branches:  
* [PDF Content Extraction](https://git.io/JcZGM)
* [Image Ranking](https://git.io/JcZGo)
* [Panel Segmentation](https://git.io/JcZG2)
* [Copy-Move Detection](https://git.io/JcZGR)
* [Provenance Analysis](https://git.io/JcZGl)

## Cite this Work
Please cite as:
> Moreira, D., Cardenuto, J.P., Shao, R. et al. SILA: a system for scientific image analysis. Nature Scientific Reports 12 (18306), 2022.
> https://doi.org/10.1038/s41598-022-21535-3

```
@article{sila,
   author = {Moreira, Daniel and Cardenuto, João Phillipe and Shao, Ruiting and Baireddy, Sriram and Cozzolino, Davide and Gragnaniello, Diego and Abd‑Almageed, Wael and Bestagini, Paolo and Tubaro, Stefano and Rocha, Anderson and Scheirer, Walter and Verdoliva, Luisa and Delp, Edward},
   title = {{SILA: a system for scientifc image analysis}},
   journal = {Nature Scientific Reports},
   year = 2022,
   number = {12},
   volume = {18306},
   pages = {1--15}
}
```
