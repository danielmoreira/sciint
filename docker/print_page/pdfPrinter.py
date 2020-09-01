"""
This is a simple script to print all pages from a PDF to a png image.
Author: Phillipe Cardenuto - May 2020

"""
import os
from glob import glob
import numpy as np
import fitz
from PIL import Image
from PIL import ImageOps
from PIL import ImageChops
import io
import shutil

class PDFPrinter(object):

    def __init__(self, input_path='.'):

        self.input_path = None
        if isinstance(input_path,list):
            self.pdf_files = []
            self.input_path = []
            if len(input_path) > 1:
                for index,pdf in enumerate(input_path):
                    if pdf.lower().endswith(".pdf"):
                        if pdf.rfind("/") == -1:
                            self.pdf_files.append(pdf)
                            self.input_path.append(".")
                        else:
                            self.pdf_files.append(pdf[pdf.rfind("/")+1:])
                            self.input_path.append(pdf[:pdf.rfind("/")])
                    else:
                        print("%s is not on *.PDF format"%pdf)
            else:
                input_path = input_path[0]
        if isinstance(input_path,str):
            if input_path.lower().endswith(".pdf"):
                self.pdf_files = []
                self.input_path = []
                pdf = input_path
                if pdf.lower().endswith(".pdf"):
                        if pdf.rfind("/") == -1:
                            self.pdf_files.append(pdf)
                            self.input_path.append(".")
                        else:
                            self.pdf_files.append(pdf[pdf.rfind("/")+1:])
                            self.input_path.append(pdf[:pdf.rfind("/")])
            else:
                self.input_path = input_path
                self.pdf_files = self.get_local_pdf()  # get all pdf from the directory
        if self.input_path is None:
            raise IOError("Input Path %s error"%input_path)
        self.img_counter = 0
        self.doc = None

    def get_local_pdf(self, input_path=None):
        if input_path is None:
            input_path = self.input_path
        pdf_files = []
        for filename in os.listdir(input_path):
            if filename.lower().endswith(".pdf"):
                pdf_files.append(filename)
        pdf_files.sort(key=str.lower)
        return pdf_files

    def print_all(self, input_path=None, out_name='.'):

        if isinstance(out_name,list):
            out_name = out_name[0]

        if not os.path.isdir(out_name):
            raise  IOError("Output %s is not a directory"%(out_name))

        if (input_path is None):
            input_path = self.input_path

        for index, pdf in enumerate(self.pdf_files):

            if isinstance(input_path,list):
                in_path =  input_path[index]
            else:
                in_path = input_path

            print(pdf)
            print(in_path + "/" + pdf)
            try:
                # insert a timeout event in case it demand too much time
                self.print_pdf(dir_path = out_name+"/"+pdf[:-4],
                               pdf = in_path+"/"+pdf)
            except:
                if (os.path.isdir((out_name + "/" + pdf[:-4]))):
                    shutil.rmtree(out_name + "/" + pdf[:-4])
                    print("Can't complete extraction of %s using pymupdf" % (pdf))
                    raise

    def print_pdf(self, pdf, dir_path=None):

        # Setting up some initial configurations
        self.img_counter = 1

        if dir_path is None:
            print_path = self.dir_path
        else:
            print_path = dir_path

        if not os.path.exists(print_path):
            os.mkdir(print_path)

        doc = fitz.open(pdf)  # open the pdf file

        try:
            for page_number, page in enumerate(doc):  # run over each page
                page_number +=1

                # rescale page
                mat = fitz.Matrix(2.0,2.0)
                
                # get figure of the page
                fig = page.getPixmap(matrix=mat,alpha=False).getImageData()

                # Transform it to Pillow format
                fig = Image.open(io.BytesIO(fig)) 
                fig = fig.convert("RGB")
                filename = f"{print_path}/page-{page_number:03d}.png"
                print(filename)
                fig.save(filename)
                
        except:
            doc.close()
            raise

        doc.close()
