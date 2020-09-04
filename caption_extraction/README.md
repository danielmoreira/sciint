# Caption Extraction
A solution to extract figure captions from PDF files.

## Requirements
* *Python 3*;
* *PDFMiner*, port to Python 3 (https://pypi.org/project/pdfminer3k/);
* Natural Language Toolkit (*NLTK*), version 3.4.5 (https://www.nltk.org/).

## Installation
We recommend the use of *pip* (https://github.com/pypa/pip).
```console
$ pip install pdfminer3k
$ pip install nltk
$ python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
```

## Simple Caption Extraction
This process prints a list containing the figure captions extracted from a given PDF file path, in the order they appear inside the document.
Each element of the list is a *(caption text, page number, bounding box)* triple.

### Usage Example
```console
$ python caption_extractor.py ./roy_2014.pdf
```

### Output
```console
('FIGURE 1. Rgt2 undergoes endocytosis and subsequent (...)        ', 4, (38.98059999999998, 124.93599999999988, 555.6398, 286.6921))
('FIGURE 2. Snf3 levels are regulated by both transcriptional (...)', 5, (38.98399999999998, 172.92399999999992, 555.6088, 266.692))
('FIGURE 3. Ubiquitination of the cytoplasmic tail domain (...)    ', 6, (39.0, 363.88520000000005, 555.6848000000001, 423.6921))
('FIGURE 4. Constitutively active Rgt2-1 and Snf3-1 glucose (...)  ', 7, (38.990999999999985, 215.38919999999996, 555.6542000000004, 300.6921))
('FIGURE 5. Signaling defective Rgt2 glucose sensor is (...)       ', 8, (39.0, 246.50400000000005, 555.632, 316.048))
('FIGURE 6. The turnover of the glucose sensors plays an (...)     ', 9, (38.99999999999997, 331.4111000000003, 291.36800000000017, 476.1551))
```

## Caption to Figure Extraction
This process takes a list of previously extracted image file paths and a single PDF file path, to extract and attribute one caption to each image.
The images must have been extracted with the latest version of Unicamp's image extractor, whose convention to name image files is:  
* p-*\<page number\>*-x0-*\<top left x\>*-y0-*\<top left y\>*-x1-*\<bottom right x\>*-x1-*\<bottom right y\>*-*\<image id\>*.ext
* E.g., *p-4-x0-38.98-y0-10.93-x1-185.63-y1-60.93-1.png*
It returns a list of *(image-file-path, caption-text)* pairs.

### Usage Example
```console
$ python caption_to_figure.py ./image_file_list.txt ./roy_2014.pdf
```

### Output
```console
('images/p-4-x0-38.98-y0-10.93-x1-185.63-y1-60.93-1.png',    'FIGURE 1. Rgt2 undergoes (...)       ')
('images/p-4-x0-190.98-y0-10.93-x1-280.63-y1-60.93-2.png',   'FIGURE 1. Rgt2 undergoes (...)       ')
('images/p-4-x0-190.98-y0-65.93-x1-280.63-y1-120.93-3.png',  'FIGURE 1. Rgt2 undergoes (...)       ')
('images/p-5-x0-38.98-y0-10.93-x1-185.63-y1-60.93-4.png',    'FIGURE 2. Snf3 levels are (...)      ')
('images/p-5-x0-190.98-y0-10.93-x1-280.63-y1-60.93-5.png',   'FIGURE 2. Snf3 levels are (...)      ')
('images/p-5-x0-190.98-y0-65.93-x1-280.63-y1-120.93-6.png',  'FIGURE 2. Snf3 levels are (...)      ')
('images/p-6-x0-38.98-y0-10.93-x1-185.63-y1-60.93-7.png',    'FIGURE 3. Ubiquitination of (...)    ')
('images/p-6-x0-190.98-y0-10.93-x1-280.63-y1-60.93-8.png',   'FIGURE 3. Ubiquitination of (...)    ')
('images/p-6-x0-190.98-y0-65.93-x1-280.63-y1-120.93-9.png',  'FIGURE 3. Ubiquitination of (...)    ')
('images/p-7-x0-38.98-y0-10.93-x1-185.63-y1-60.93-10.png',   'FIGURE 4. Constitutively active (...)')
('images/p-7-x0-190.98-y0-10.93-x1-280.63-y1-60.93-11.png',  'FIGURE 4. Constitutively active (...)')
('images/p-7-x0-190.98-y0-65.93-x1-280.63-y1-120.93-12.png', 'FIGURE 4. Constitutively active (...)')
('images/p-8-x0-38.98-y0-10.93-x1-185.63-y1-60.93-13.png',   'FIGURE 5. Signaling defective (...)  ')
('images/p-8-x0-190.98-y0-10.93-x1-280.63-y1-60.93-14.png',  'FIGURE 5. Signaling defective (...)  ')
('images/p-8-x0-190.98-y0-65.93-x1-280.63-y1-120.93-15.png', 'FIGURE 5. Signaling defective (...)  ')
('images/p-9-x0-38.98-y0-10.93-x1-185.63-y1-60.93-16.png',   'FIGURE 6. The turnover of the (...)  ')
('images/p-9-x0-190.98-y0-65.93-x1-280.63-y1-120.93-17.png', 'FIGURE 6. The turnover of the (...)  ')
```

## Problems or Questions?
Please contact Daniel Moreira (dhenriq1@nd.edu).
