# Scientific Integrity Dataset - Content Segmentation
### Version 8, last revision time: July 31, 2021

This repository presents the subset of Scientific Integrity Dataset designed to benchmark the *Content Segmentation* task.

*All collected data are public available and were collected from scientific publishers or trustworthy repositories (e.g., [PubMed](https://pubmed.ncbi.nlm.nih.gov/)).*



## <a name="json-file">Dataset Annotation</a>

The [Content Segmentation Dataset](panel-segmentation-annotation.json) is a JSON-based file containing 303 annotated scientific figures indicating the position of its panels.

The annotation includes:

- Position of each Figure's panel.

- Link to online version of the Figure.

- Path to the ground-truth (binary mask) after unzipping [gt.zip](gt.zip)

  

If you are not a computer science expert, we recommend using a [JSON viewer](http://jsonviewer.stack.hu/) to analyze these annotations.

<details>
<summary>Annotation JSON Structure - click to expand</summary><p>

```python
                                                                                  # Field Explanation #
                                                                   ##########################################################
{
    "<Figure-ID>": {                             # Figure ID
        "figname": "<figure-name>",              # Figure label (e.g, "fig1")
        "panels-location": {                     # Location of each panel within the figure
            "<panel-ID>": {                      # Panel ID. This ID is unique only for this Figure
                "x0": <x0>,                      # x0 position
                "y0": <y0>,                      # y0 position               
                "x1": <x1>,                      # x1 position
                "y1": <y1>                       # y1 position
            },
        },
        "gt-path": "<path-to-ground-truth>",     # Ground-truth binary mask, after unzipping the gt.zip
        "fig-path": "<figure-save-path>",        # Path to save the figure after downloading
        "fig-url": "<figure-url>"                # Figure Online version
    }
}
```



</p>
</details>

The zip files located at this repository is a copy of the data used during our experiments.

-  [gt.zip](gt.zip) contains the annotated binary-masks of each figure used on the *Content Extraction* experiment. This file is *open access*.
- [figures.zip](figures.zip) contains all figures used during the *Content Extraction* experiment. If you want to use these data for academic purposes, please contact [Daniel Moreira](daniel.moreira@nd.edu).
