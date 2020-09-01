import argparse

from pdfPrinter import PDFPrinter
from pdfPrinter import *

parser = argparse.ArgumentParser(prog='print.py', description="This Program prints each page of a PDF to apng image.", formatter_class=argparse.MetavarTypeHelpFormatter)
parser.add_argument("--input_path","-i", required=True, type=str ,nargs='+',
                    help="Path to the PDF input directory.\
                    \n You can also input the name of each PDF.")
parser.add_argument("--output_path","-o", type=str ,nargs=1,default=".",
                    help="Path to the output directory.\nDefault \"./\"")
args = vars(parser.parse_args())

def main():
    input_path = args['input_path']
    outpath = args['output_path']

    printer = PDFPrinter(input_path=input_path)
    printer.print_all(out_name=outpath)
    

if __name__ == "__main__":
    main()
