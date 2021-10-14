# Scientific Integrity Dataset - PDF Content Extraction
### Version 8, last revision time: July 31, 2021

This repository presents the subset of Scientific Integrity Dataset designed to benchmark *PDF content extraction*

*All collected data are public available and were collected from scientific publishers or trustworthy repositories (e.g., [PubMed](https://pubmed.ncbi.nlm.nih.gov/)).*



## <a name="json-file">Dataset Annotation</a>

The [PDF Content Extraction Dataset](pdf-content-extraction-annotation.json) is a JSON-based file containing annotation of the Figures and Caption related to a scientific paper. The annotation includes:
- Position of the Figures within the paper's PDF.
- Link to online version of the Figure.
- Full Caption text of the Figure.
- Link to online version of the PDF.

If you are not a computer science expert, we recommend using a [JSON viewer](http://jsonviewer.stack.hu/) to analyze these annotations.

<details>
<summary>Annotation JSON Structure - click to expand</summary><p>

```python
                                                                                  # Field Explanation #
                                                                   ##########################################################
{
    "PDF-<ID>": {                                                  # PDF ID
        "pdf-path": "<pdf_save_path>",                             # Path to save the PDF after downloading
        "pdf-figures": {                                           # All annotated figures from the PDF
            "fig1": {                                              # Figure label
                "bbox-loc": {                                      # Position of the figure within the PDF document
                    "p": "<page>",                                 # PDF Page which the figure is presented
                    "y0": "<y0>",                                  # y0 position
                    "y1": "<y1>",                                  # y1 position
                    "x0": "<x0>",                                  # x0 position
                    "x1": "<x1>"                                   # x1 position
                },
                "figure-path": "<figure_save_path>",               # Path to save the figure after downloading
                "caption-path": "<caption_save_path>",             # Path to save the caption
                "caption-txt": "<content>",                        # Full caption text related to the figure
                "figure-url": "<fig-url>"                          # Figure Online version
            },
            "pdf-url": "<pdf-url>"                                 # PDF Online version
        }
    }
}
```



</p>
</details>

The file [pdf-content-extraction-data.zip](pdf-content-extraction-data.zip
) is a copy of all downloaded Figures and PDFs needed to replicate the *PDF Content Extraction* experiments. If you want to use these data for academic purposes and want to have access to its password, please contact [Daniel Moreira](https://github.com/danielmoreira/sciint/blob/provenance-analysis/daniel.moreira@nd.edu).





