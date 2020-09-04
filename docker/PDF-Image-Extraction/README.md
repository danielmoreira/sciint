# PDF-Image-Extraction
Extract images from PDFs using the PDF stream data

-----

## Instructions

It has 3 operations mode:

* <u>SAFE MODE</u>: Extract images that has xref, considering alpha layers and stencil  mask  but not  data corruption
*  <u>NORMAL MODE</u>: Extract images that has xref and have being duplicate. It considered alpha layers, stencil masks, images positions on page, data corruption.  But, it could fails if the image is too small.
* <u>UNSAFE MODE</u>: Extract all images , even without xref, and it has no warranty.(NOT RECOMMENDED).

​     All images are saved in <*.png> format.

The sample PDF was downloaded from https://www.hbp.com/resources/SAMPLE%20PDF.pdf

# Quick Start

Environment used to run the project:

- Python 3

  - pymupdf, pillow, numpy

  There is a requirements.txt that can help with the environment.

```bash
pip install -r requirements.txt
```

####  Extraction

Run `python extraction.py -i .` to extract figures of all PDFs at "./" directory.

### Usage

```bash
extract.py [-h] --input_path str [str ...] [--output_path str] [--mode mode]
```

* optional arguments:
  * --input_path str [str ...], -i str [str ...]
    ​                        Path to the PDF input directory. You can also input  the name of each PDF.
  * --output_path str, -o str
    ​                        Path to the output directory. Default "./"
  * --mode mode, -m mode  Operation mode of the extraction. DEFAULT: (NORMAL)
* The output figure name is different for each mode:
  * <u>SAFE:</u> p-< page>-<count_number>.png
  * <u>NORMAL:</u>  p-< page>-x0-<x0_coordinate>-y0-<y0_coordinate>-x1-<x1_coordinate>-y1-<y1_coordinate> -<count_number>.png . The coordinates are related with the PDF page.
  * <u>UNSAFE:</u> p-< page>-<count_number>.png



##### AUTHOR

João Phillipe Cardenuto ,

```
				 UNICAMP (University of Campinas) RECOD Lab
```

You may send me an email to any kind of discussion or collaboration:  phillipe.cardenuto@ic.unicamp.br
