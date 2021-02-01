# Scientific Integrity Dataset
### Version 8, last edit time: Jan 30, 2021

#### Compared with dataset version 7, we reorganize the dataset and collect more papers.

Dataset contains detail information of 988 papers and data_annotation.xls.
``data_annotation.xls`` contains basic information of each collected paper.  

Columns in the ``data_annotation.xls``:   
       Directory: the directory name of the collected paper  
       DOI	  
       Title: the full name of the collected paper   
       Article Link	  
       Authors	  
       Cited By: the number of the cited papers	  
       Copyright	  
       Publication Source  
       Published Date  
       Received Date	  
       Accepted Date	  
       Publisher	  
       Has Retraction/Correction	  
       Reason for Retreaction/Correction:  
              Abbreviations:  
                     FL  Falsification of Image  
                     FB  Fabrification of Image  
                     PL  Plagriarism of Image  
       Retraction/Correction Notice Available	  
       Retraction/Correction DOI	  
       Source Image Available	  
       Supplmentary Material Available   

For each collected paper:
```bash

Directory Name (named as doi by replacing '/' with '_')
├── article
│   └── paper.pdf
├── figures 
│   ├── collected images: named in figx.xxx format
│   └── captions: named same as the corresponding image in txt format, like figx.txt
├── figures-gt (if available)
│   ├── cmfd-map 
│   │   └── ground-truth images used for copy-move detection: named same as the corresponfing query image
│   └── panels 
│       └── ground-truth images used for panel extraction: named same as the corresponfing query image
├── figures-panels (if available)
│   └── panel images: for image figx.xxx, its subpanel images are named in figx_xxx.xxx format
├── infos 
│   └── info.json (metadata of the collected paper)
├── retraction-correction
│   ├── notice.pdf
│   └── notice.txt
└── supplementary (if available)
    └── supplemental material

```
