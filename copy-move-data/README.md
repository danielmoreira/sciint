# Scientific Integrity Dataset - Copy-Move Detection
### Version 8, last revision time: July 31, 2021

This repository presents the subset of Scientific Integrity Dataset designed to benchmark the *Copy-Move Detection* task.

*All collected data are public available and were collected from scientific publishers or trustworthy repositories (e.g., [PubMed](https://pubmed.ncbi.nlm.nih.gov/)).*



## <a name="json-file">Dataset Annotation</a>

The [Copy-Move Detection Dataset](copy-move-annotation.json) is a JSON-based file containing 180 annotated scientific figures indicating the position of a reported inappropriate copy-move image region.

The annotation includes:

- The binary-mask path for each annotated figure

- Reported Figure online version

- Retraction/Correction Notice URL related to the reported figure

- Scientific Article DOI

  

If you are not a computer science expert, we recommend using a [JSON viewer](http://jsonviewer.stack.hu/) to analyze these annotations.

<details>
<summary>Annotation JSON Structure - click to expand</summary><p>

```python
                                                             # Field Explanation #
                                           ##########################################################
{
    <case-id>: {                           # Figure ID
        "figure-path": <save_path_figure>, # Path to save the figure after downloading
        "doi": <doi>,                      # Article DOI related to the reported figure
        "figure-url": <figure-url>,        # Figure Online version
        "notice-url": <notice-url>,        # Retraction/Correction Online version
        "gt-path": <path-to-ground-truth>  # Ground-truth binary mask path, after unzipping the gt.zip
    }
}

```



</p>
</details>

The zip files located at this repository is a copy of the data used during our experiments.

-  [gt.zip](gt.zip) contains the annotated binary-masks of each figure used on the *Copy-Move Detection* experiment. This file is *open access*.
- [figures.zip](figures.zip) contains all figures used during the *Copy-Move Detection* experiment. If you want to use these data for academic purposes, please contact [Daniel Moreira](https://github.com/danielmoreira/sciint/blob/provenance-analysis/daniel.moreira@nd.edu).
