"""
extract.py:  This Program extract images from raw PDF data, avoiding mask and alpha channels.
                It has 3 operations mode:
                    |> SAFE MODE: Extract images that has xref, considering alpha layers and stencil mask
                                    but not  data corruption

                    |> NORMAL MODE: Extract images that has xref and have being duplicate. It considered alpha layers,
                                    stencil masks, images positions on page, data corruption.
                                    But it could fails if the image is too small.

                    |> UNSAFE MODE: Extract all images , even without xref, and it has no warranty.(NOT RECOMMENDED).
                All images are saved in <*.png> format.

Author:        Joao Phillipe Cardenuto - University of Campinas (UNICAMP)
Email:         phillipe.cardenuto@ic.unicamp.br
Be free to send me an email or contact me to report any issue about this program.
"""
import argparse

from pdfExtractor import PDFExtractor
from pdfExtractor import *

parser = argparse.ArgumentParser(prog='extract.py', description="This Program extract images from raw PDF data. All images are saved in <*.png> format.", formatter_class=argparse.MetavarTypeHelpFormatter)
parser.add_argument("--input_path","-i", required=True, type=str ,nargs='+',
                    help="Path to the PDF input directory.\
                    \n You can also input the name of each PDF.")
parser.add_argument("--output_path","-o", type=str ,nargs=1,default=".",
                    help="Path to the output directory.\nDefault \"./\"")
parser.add_argument("--mode", "-m", type=str ,nargs=1, default='normal', metavar=("mode"),
                    help="Operation mode of the extraction. DEFAULT: (NORMAL)")                    
args = vars(parser.parse_args())

def main():
    input_path = args['input_path']
    outpath = args['output_path']
    if type(args['mode']) == list:
        mode = args['mode'][0]
    else:
        mode = args['mode']
    extractor = PDFExtractor(input_path=input_path)
    extractor.extract_all(out_name=outpath,mode=mode)
    

if __name__ == "__main__":
    main()
