#!/bin/bash
{ time python3 /work/facade/pdf_image_extractor.py -i "/INPUT/$PDF" -o "/OUTPUT/" ; } 2> \
            /OUTPUT/"$PDF_NAME"/extraction_time.txt

