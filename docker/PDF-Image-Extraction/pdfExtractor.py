"""
pdfExtractor.py:  This Program extract images from raw PDF data, avoiding mask and alpha channels.
                It has 3 operations mode:
                    |> SAFE MODE: Extract images that has xref, considering alpha layers and stencil mask
                                    but not  data corruption

                    |> NORMAL MODE: Extract images that has xref and have being duplicate. It considered alpha layers,
                                    stencil masks, images positions on page, data corruption.
                                    But it could fails if the image is too small.

                    |> UNSAFE MODE: Extract all images , even without xref, and it has no warranty.(NOT RECOMMENDED).
                All images are saved in <*.png> format.

Author:        Joao Phillipe Cardenuto - University of Campinas (UNICAMP)
Email:         phillipe.cardenuto@gmail.com
Be free to send me an email or contact me to report any issue about this program.

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
import signal

#Timeout event
def handler_timeout(signum,frame):
    raise TimeoutError("TIMEOUT!")

class ImageEmbedded:
    """
     This class organizes all data from a pdf embedded image.
    """

    def __init__(self, xref=None, bbox=None, image_setting=None, width=None, height=None,
                 alt_colorspace=None, doc=None):
        """

        :param xref: xref pdf image
        :param bbox: location of the image in a PDF page
        :param image_setting: dict containing image's info.
        :param width: image width
        :param height: image height
        :param alt_colorspace: image colorspace
        :param doc: image PDF doc

        """

        if image_setting:
            self.xref = xref
            self.ext = image_setting["ext"]
            self.smask = image_setting['smask']
            self.colorspace = image_setting['colorspace']
            self.image = None
            configs = doc._getXrefString(xref).split("/")

            if "Filter" in configs:
                self.filter = configs[configs.index("Filter") + 1]
            else:
                self.filter = None

            self.width = image_setting['width']
            self.height = image_setting['height']
            self.alt_colorspace = alt_colorspace

        else:
            self.xref = None
            self.ext = None
            self.smask = None
            self.colorspace = None
            self.image = None
            self.filter = None
            self.width = width
            self.height = height
            self.alt_colorspace = alt_colorspace

        if bbox:
            (x0, y0, x1, y1) = bbox

            self.bbox = fitz.Rect(bbox)
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1
        else:
            self.bbox = None

    def copy(self):
        """
        Make a copy from a ImageEmbedded obj
        :return  copy from the ImageEmbedded with xref = No
        """
        copy_obj = ImageEmbedded(None, None)
        copy_obj.ext = self.ext
        copy_obj.smask = self.smask
        copy_obj.colorspaces = self.colorspace
        copy_obj.image = self.image
        copy_obj.filter = self.filter
        copy_obj.width = self.width
        copy_obj.height = self.height
        copy_obj.alt_colorspace = self.alt_colorspace  # alternative color
        copy_obj.bbox = fitz.Rect(self.bbox)
        return copy_obj

    def has_alpha(self):
        if self.smask:
            return True
        return False


def get_rectangles_points(bbox):
    p0 = fitz.Point(bbox.x0, bbox.y0)
    p1 = fitz.Point(bbox.x1, bbox.y0)
    p2 = fitz.Point(bbox.x0, bbox.y1)
    p3 = fitz.Point(bbox.x1, bbox.y1)
    return p0, p1, p2, p3


def check_overlap(bboxi, bboxj, distance=1, distance_bbox=0.001):
    """
    Check if there is overlapping between the input bbox
    :param bboxi: <tuple>
    :param bboxj: <tuple>
    :param distance: acceptable coordinate distances
    :param distance_bbox: acceptable bbox distances
    :return: <bool>
    """
    # P0  #### P1
    #     |  |
    #  P2 #### P3

    p0, p1, p2, p3 = get_rectangles_points(bboxi)
    q0, q1, q2, q3 = get_rectangles_points(bboxj)
    # 2 differents figures is locates at the exactly location

    if p0.distance_to(q0) == 0 and p1.distance_to(q1) == 0 \
            and p2.distance_to(q2) == 0 and p3.distance_to(q3) == 0:
        return False
    if bboxi in bboxj or bboxj in bboxi:
        return True

    if p1.distance_to(q0) < distance and \
            p3.distance_to(q2) < distance:
        return True

    if p0.distance_to(q1) < distance and \
            p2.distance_to(q3) < distance:
        return True

    if p0.distance_to(q2) < distance and \
            p1.distance_to(q3) < distance:
        return True

    if p2.distance_to(q0) < distance and \
            p3.distance_to(q1) < distance:
        return True

    if p1.distance_to(bboxj) < distance_bbox and \
            p3.distance_to(bboxj) < distance_bbox and \
            (p1.distance_to(q0) < distance_bbox or p3.distance_to(q2) < distance_bbox):
        return True

    if p0.distance_to(bboxj) < distance_bbox and \
            p2.distance_to(bboxj) < distance_bbox and \
            (p0.distance_to(q1) < distance_bbox or p2.distance_to(q3) < distance_bbox):
        return True

    if p0.distance_to(bboxj) < distance_bbox and \
            p1.distance_to(bboxj) < distance_bbox and \
            (p0.distance_to(q2) < distance_bbox or p1.distance_to(q3) < distance_bbox):
        return True

    if p2.distance_to(bboxj) < distance_bbox and \
            p3.distance_to(bboxj) < distance_bbox and \
            (p2.distance_to(q0) < distance_bbox or p3.distance_to(q1) < distance_bbox):
        return True

    return False


class PDFExtractor(object):
    """
    Extracts images from PDF, using the object carried by a raw pdf
     pymupdf (https://pymupdf.readthedocs.io/en/latest/) [DEFAULT]
            If the colorspace of the image is CMYK it will be converted to RGB. No mask images will be considered
            Small images less than (10,10) will be ignored.
            All images are saved as *.png extension.

    You can choose three operation mode:
        |> SAFE MODE: Extract images that has xref.
        |> NORMAL MODE(DEFAULT): Extract images that has xref and have being duplicate and avoid data corruption.
        |> UNSAFE MODE: Extract all images. (NOT RECOMMENDED)
    Initializes the object with all files with extension [.pdf,.PDF]

    """
    operations_mode = ['safe', 'normal', 'unsafe']

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

    def extract_all(self, input_path=None, out_name='.', mode='normal'):
        """
        Extracts all images from a list of PDF initialized with get_names_pdf().
        By default , the operation mode is set by pymupdf

        Parameters
        ----------
        dir_path: string
            Path to directory of the pdf.
        out_name: string
            Name used in the beginning of a File. Just in pdfimages mode
        mode: string
            Operation execution mode
        """
        if not mode in PDFExtractor.operations_mode:
            raise ValueError(mode)
        if isinstance(out_name,list):
            out_name = out_name[0]
        if not os.path.isdir(out_name):
            raise  IOError("Output %s is not a directory"%(out_name))
        if (input_path is None):
            input_path = self.input_path
        if (mode == 'normal'):
            for index, pdf in enumerate(self.pdf_files):
                if isinstance(input_path,list):
                    in_path =  input_path[index]
                else:
                    in_path = input_path
                print(pdf)
                print(in_path + "/" + pdf)
                try:
                    # insert a timeout event in case it demand too much time
                    signal.signal(signal.SIGALRM, handler_timeout)
                    signal.alarm(600)
                    self.normal_mode(dir_path=out_name + "/" + pdf[:-4], pdf=in_path + "/" + pdf)
                except KeyboardInterrupt:
                    signal.alarm(0)
                    raise
                except TimeoutError as te:
                    try:
                        print("%s is taking too much time"%(pdf))
                        print("TRYING THE SAFE MODE...")
                        shutil.rmtree(out_name + "/" + pdf[:-4])
                        self.safe_mode(dir_path=out_name + "/" + pdf[:-4], pdf=in_path + "/" + pdf)
                    except:
                        if (os.path.isdir((out_name + "/" + pdf[:-4]))):
                            shutil.rmtree(out_name + "/" + pdf[:-4])
                            print("Can't complete extraction of %s using pymupdf" % (pdf))
                except:
                    try:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, handler_timeout)
                        signal.alarm(600)
                        print("%s can't be extract with normal mode"%pdf)
                        print("TRYING THE SAFE MODE...")
                        if (os.path.isdir((out_name + "/" + pdf[:-4]))):
                            shutil.rmtree(out_name + "/" + pdf[:-4])
                            self.safe_mode(dir_path=out_name + "/" + pdf[:-4], pdf=in_path + "/" + pdf)
                    except:
                        if (os.path.isdir((out_name + "/" + pdf[:-4]))):
                            shutil.rmtree(out_name + "/" + pdf[:-4])
                            print("Can't complete extraction of %s using pymupdf" % (pdf))
                finally:
                    signal.alarm(0)
                    self.posprocessing_extraction(dir_path=out_name + "/" + pdf[:-4])

        elif (mode == 'safe'):
             for index, pdf in enumerate(self.pdf_files):
                if isinstance(input_path,list):
                    in_path =  input_path[index]
                else:
                    in_path = input_path
                print(pdf)
                try:
                    self.safe_mode(dir_path=out_name + "/" + pdf[:-4], pdf=in_path + "/" + pdf)
                except:
                    print("Can't complete extraction of %s using pymupdf safe mode" % (pdf))
                    raise
        else:
             for index, pdf in enumerate(self.pdf_files):
                if isinstance(input_path,list):
                    in_path =  input_path[index]
                else:
                    in_path = input_path
                print(pdf)
                try:
                    self.unsafe_mode(dir_path=out_name + "/" + pdf[:-4], pdf=in_path + "/" + pdf)
                except:
                    print("Can't extract using pymupdf unsafe mode")
                    raise

        print("Extraction Done!")

    def is_sigle_color(self, img_path):
        """
        Check if image has only one color
        :param img_path: path to image
        :return: <bool>
        """
        img = Image.open(img_path)
        extrema = img.getextrema()
        if len(extrema) >=3:
            for band_extrema in extrema:
                if band_extrema[0] != band_extrema[1]:
                    return False
            return True
        else:
            if extrema[0] != extrema[1]:
                return False
        return True
    def is_equal_imgs(self, img_path_i, img_path_j):
        """
        Check if the input image are equal
        :param img_path_i: path to image i
        :param img_path_j: path to image j
        :return: <bool>
        """
        img_i = Image.open(img_path_i)
        img_j = Image.open(img_path_j)
        if img_i.size != img_j.size:
            return False, None
        img_i.load()
        img_j.load()
        if img_i.mode != img_j.mode:
            # Image is RGB and it's copy GrayScale. Probably is a alpha layer. Destoy the alpha one
            if img_j.mode == 'L':
                return True, img_path_j
            if img_i.mode == 'L':
                return True, img_path_i
        if img_i.mode == img_j.mode:
            # Image is GrayScale and also it's copy. Probably has alpha layer. Destoy the alpha one
            if img_i.mode == 'L':
                last = [img_path_i,img_path_j]
                last.sort(key=lambda x: int(x[x.rfind("-"):-4]), reverse=True)
                last = last[1]
                return True, last
            else:
                # Image is not GrayScale and have at same location a copy, probably is a PDF damaged
                if ImageChops.difference(img_i, img_j).getbbox() is None:
                    return True, img_path_i
        return False, None

    def isclose_infos(self,img_i,img_j):
        for index in range(0,5):
            if not np.isclose(img_i[index],img_j[index]):
                return False
        return True

    def posprocessing_extraction(self,dir_path):
        """
        Eliminate duplicate and single color images
        :param dir_path: path to extracted images
        """
        imgs_names = glob(dir_path+"/*.png")
        imgs_names.sort(key=lambda x: int(x[x.rfind("-"):-4]), reverse=True)
        #imgs_infos has (page,x0,y0,x1,y1,index);
        # index
        imgs_infos = [ (int(img[img.find("/p-")+3:img.find("-x0-")]),
                       float(img[img.find("-x0-")+4:img.find("-y0-")]),
                       float(img[img.find("-y0-") + 4:img.find("-x1-")]),
                       float(img[img.find("-x1-") + 4:img.find("-y1-")]),
                       float(img[img.find("-y1-") + 4:img.rfind("-")]),
                       index) for index,img in enumerate(imgs_names)]
        while imgs_infos != []:
            img_i = imgs_infos.pop(0)
            if self.is_sigle_color(imgs_names[img_i[5]]):
                os.remove(imgs_names[img_i[5]])
                continue

            for index,img_j in enumerate(imgs_infos):
                if  self.isclose_infos(img_i,img_j):
                    delete, delete_image_name = self.is_equal_imgs(imgs_names[img_i[5]],imgs_names[img_j[5]])
                    if delete:
                        if delete_image_name == imgs_names[img_j[5]]:
                            imgs_infos[index] = img_i
                        os.remove(delete_image_name)
                        break



    def unsafe_mode(self,pdf, dir_path=None):
        """
        Use pymupdf to extract all images from a PDF file. This method is not robust to several types of image corruption.
        In this unsafe mode all image of the pdf are extracted not using any warranty of duplicated xrefences
        Due to it duplicate images can appears.
        The output directory is the same as the dir_path.
        Small images less then (10,10) will be ignored
        Parameters
        pdf: <string> PDF file name.
        dir_path: <string> Path to directory of the pdf.
        """
        # Setting up some initial configurations

        self.img_counter = 1
        if dir_path is None:
            extraction_path = self.dir_path
        else:
            extraction_path = dir_path

        if not os.path.exists(extraction_path):
            os.mkdir(extraction_path)

        self.doc = fitz.open(pdf)  # open the pdf file
        try:
            for page in range(len(self.doc)):  # run over each page
                # this method get a dictionary with all texts and image from a page
                page_contents = self.doc[page].getText("dict")
                all_image_from_page = [t for t in page_contents['blocks'] if t['type'] == 1]
                for index in range(len(all_image_from_page)):
                    img = all_image_from_page[index]['image']
                    img = Image.open(io.BytesIO(img))
                    img.load()
                    if img.width < 10 or img.height < 10:
                        continue
                    name = "%s/p-%s-%s.png" % (extraction_path, str(page + 1), self.img_counter)
                    img.save(name)
                    self.img_counter += 1
        except:
            self.doc.close()
            raise

        self.doc.close()

    def safe_mode(self, pdf, dir_path=None):
        """
        Use pymupdf to extract all images from a PDF file. This method is not robust to several types of image corruption.
        This method extract all xreferred image from the PDF.
        The output directory is the same as the dir_path.
        Small images less then (10,10) will be ignored
        Parameters
        pdf: <string> PDF file name.
        dir_path: <string> Path to directory of the pdf.
        """
        # Setting up some initial configurations
        self.img_counter = 1
        if dir_path is None:
            extraction_path = self.dir_path
        else:
            extraction_path = dir_path

        if not os.path.exists(extraction_path):
            os.mkdir(extraction_path)

        self.doc = fitz.open(pdf)  # open the pdf file
        xrefs_checked = []  # List of xrefs object used in extraction ( it can have duplicated xrefs,
        #  if the a same image twice on the same page.
        try:
            for page in range(len(self.doc)):  # run over each page
                for img in self.doc.getPageImageList(page):
                    # img is a list with [xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter]
                    xref = img[0]
                    if xref in xrefs_checked:
                        continue
                    smask = img[1]
                    if smask != 0:  # has stencil mask
                        pix1 = fitz.Pixmap(self.doc, xref)
                        pix2 = fitz.Pixmap(self.doc, smask)
                        pix = fitz.Pixmap(pix1)
                        pix.setAlpha(pix2.samples)
                        if self.write_img(pix, "%s/p-%s-%s.png" % (extraction_path, str(page + 1), self.img_counter)):
                            self.img_counter += 1
                            xrefs_checked.append(xref)
                        pix = None
                        pix1 = None
                        pix2 = None

                    else:  # does not has stencil mask
                        pix = fitz.Pixmap(self.doc, xref)
                        if self.write_img(pix, "%s/p-%s-%s.png" % (extraction_path, str(page + 1), self.img_counter),
                                          img[5]):
                            self.img_counter += 1
                            xrefs_checked.append(xref)
                        pix = None

        except:
            self.doc.close()
            raise

        self.doc.close()

    def normal_mode(self, pdf, dir_path=None):
        """
        Use pymupdf to extract all images from a PDF file. This method is robust to several types of image corruption.
        The output directory is the same as the dir_path.
        Small images less then (10,10) will be ignored
        Parameters
        ----------
        pdf: <string> PDF file name.
        dir_path: <string> Path to directory of the pdf.
        """
        # Setting up some initial configurations
        self.img_counter = 1
        if dir_path is None:
            extraction_path = self.dir_path
        else:
            extraction_path = dir_path

        if not os.path.exists(extraction_path):
            os.mkdir(extraction_path)

        self.doc = fitz.open(pdf)  # open the pdf file
        xrefs_checked = []  # List of xrefs object used in extraction ( it can have duplicated xrefs,
        #  if the a same image twice on the same page.
        try:
            # Run through every page and extract all images.
            for page in range(len(self.doc)):
                # this method get a dictionary with all texts and image from a page
                page_contents = self.doc[page].getText("dict")
                all_image_from_page = [t for t in page_contents['blocks'] if t['type'] == 1]
                xreferred_image_list = []

                for img in self.doc.getPageImageList(page):
                    # img is a list with [xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter]
                    xref = img[0]
                    extract_img = self.doc.extractImage(xref)
                    # If xref was alredy referreced in this page continue
                    if xref in xrefs_checked:
                        continue
                    # If image has alpha, we don't assembly it
                    if extract_img["smask"] > 0:
                        pix_img = fitz.Pixmap(self.doc,xref)
                        if pix_img.colorspace.name == 'DeviceCMYK':
                            pix_img = fitz.Pixmap(fitz.csRGB, pix_img)  # converting to csRGB
                        aux_img = pix_img.getPNGData()
                        pix_img = None
                        found = False
                        for p_img in all_image_from_page:
                            if p_img['image'] == aux_img:
                                xreferred_image_list.append(
                                    ImageEmbedded(xref, p_img['bbox'], extract_img, alt_colorspace=img[5], doc=self.doc))  #
                                xrefs_checked.append(xref)
                                found = True
                                aux_img = None
                                break
                        if found:
                            continue

                        xreferred_image_list.append(
                            ImageEmbedded(xref, None, extract_img, alt_colorspace=img[5], doc=self.doc))  #
                        xrefs_checked.append(xref)
                        continue

                    pix = fitz.Pixmap(self.doc, xref)
                    if pix.colorspace == None:
                        continue

                    # Here we are getting the location of imgs bbox.
                    #  Note that we allow with this duplications on xrefs_checked, if the image same image appears twice.
                    index = 0
                    while (index < len(all_image_from_page)):
                        if extract_img['image'] == all_image_from_page[index]['image']:
                            secure_to_add = True

                            for obj in xreferred_image_list:
                                # if the new image has the same location of one already added, and the same image.
                                # It is not safe to add on the images list.
                                if all_image_from_page[index]['bbox'] == obj.bbox:
                                    if obj.image == all_image_from_page[index]['image']:
                                        secure_to_add = False
                                        break
                                    if obj.smask:
                                        secure_to_add = False
                                        break

                            # Image is not a Stencil Mask and has big sizes ( not corrupted), we have to isolate it.
                            # It could be a copy-paste.
                            if xref in xrefs_checked:
                                if extract_img['width'] > 30 and extract_img['height'] > 30:
                                    img[5] = "Isolate%d" % index

                            if secure_to_add:
                                xreferred_image_list.append(
                                    ImageEmbedded(xref, all_image_from_page[index]['bbox'], extract_img,
                                                  alt_colorspace=img[5],
                                                  doc=self.doc)
                                )
                                xrefs_checked.append(xref)
                                all_image_from_page.pop(index)
                                continue
                        # index loop
                        index += 1
                # If we didn't found any xreferred image, we insert all images from page
                if len(xreferred_image_list) == 0 and len(all_image_from_page) > 0:
                    index = 0
                    for img in self.doc.getPageImageList(page):
                        xref = img[0]
                        if xref in xrefs_checked:
                            continue
                        extract_img = self.doc.extractImage(xref)
                        if extract_img["smask"] > 0:  # Image has alpha, we don't assembly it
                            xreferred_image_list.append(ImageEmbedded(xref, None, extract_img, alt_colorspace=img[5],
                                                                      doc=self.doc))
                            index += 1
                            continue
                        else:

                            xreferred_image_list.append(
                                ImageEmbedded(xref, all_image_from_page[index]['bbox'], extract_img,
                                              alt_colorspace=img[5],
                                              doc=self.doc))
                            index += 1

                del all_image_from_page

                # If we found  more then one images, we check if there is overlapping on their location
                if len(xreferred_image_list) > 1:
                    overlap_set = self.build_overlap_set(xreferred_image_list)
                    overlap_list = [list(o_set) for o_set in overlap_set]

                    for list_ in overlap_list:

                        embedded_overlapping_figures = [xreferred_image_list[i] for i in list_]

                        if embedded_overlapping_figures[0].has_alpha():
                            # We're considering that imgs with alpha layer doesn't corrupt
                            # Setting Pixmap with Stencil Mask
                            figure = embedded_overlapping_figures[0]
                            pix1 = fitz.Pixmap(self.doc, figure.xref)
                            pix2 = fitz.Pixmap(self.doc, figure.smask)
                            pix = fitz.Pixmap(pix1)
                            pix.setAlpha(pix2.samples)
                            if figure.bbox:

                                file_name = "%s/p-%s-x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" %\
                                        (extraction_path, str(page + 1), figure.x0, figure.y0,
                                         figure.x1, figure.y1, self.img_counter)
                            else:
                                file_name = "%s/p-%s-x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                            (extraction_path, str(page + 1), 0, 0, 0, 0, self.img_counter)

                            if self.write_img(pix, file_name):
                                self.img_counter += 1
                            pix = None
                            pix1 = None
                            pix2 = None
                        else:
                            if len(embedded_overlapping_figures) > 1:  # Image has overlap
                                if self.assembly_image(embedded_overlapping_figures,
                                                       "%s/p-%s-" % (extraction_path, str(page + 1))):
                                    self.img_counter += 1
                            else:  # Image doesn't has overlap
                                figure = embedded_overlapping_figures[0]
                                if figure.xref:  # If figure was xreferred get it, using the fitz method
                                    pix = fitz.Pixmap(self.doc, embedded_overlapping_figures[0].xref)
                                    file_name = "%s/p-%s-x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                                (extraction_path, str(page + 1), figure.x0, figure.y0,
                                                 figure.x1, figure.y1, self.img_counter)
                                    if self.write_img(pix, file_name, figure.alt_colorspace):
                                        self.img_counter += 1
                                    pix = None
                                else:  # If the figure wasn't xreferred, it is already on the field image of the ImageEmbedded obj
                                    if figure.image.size[0] > 10 and figure.image.size[1] > 10:
                                        file_name = "%s/p-%s-x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                                    (extraction_path, str(page + 1), figure.x0, figure.y0,
                                                     figure.x1, figure.y1, self.img_counter)
                                        figure.image.save(file_name)
                                        self.img_counter += 1
                                        figure = None

                elif xreferred_image_list != []:
                    figure = xreferred_image_list[0]
                    if figure.has_alpha():  # With all our observation, any img with alpha band  was not cropped
                        pix1 = fitz.Pixmap(self.doc, figure.xref)
                        pix2 = fitz.Pixmap(self.doc, figure.smask)
                        pix = fitz.Pixmap(pix1)
                        pix.setAlpha(pix2.samples)
                        if figure.bbox:
                            file_name = "%s/p-%s-x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                        (extraction_path, str(page + 1), figure.x0, figure.y0,
                                         figure.x1, figure.y1, self.img_counter)
                        else:
                            file_name = "%s/p-%s-x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                        (extraction_path, str(page + 1), 0, 0, 0, 0, self.img_counter)
                        if self.write_img(pix, file_name, figure.alt_colorspace):
                            self.img_counter += 1

                    else:

                        pix = fitz.Pixmap(self.doc, figure.xref)
                        if figure.bbox:
                            file_name = "%s/p-%s-x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                    (extraction_path, str(page + 1), figure.x0, figure.y0,
                                     figure.x1, figure.y1, self.img_counter)
                        else:
                            file_name = "%s/p-%s-x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                        (extraction_path, str(page + 1), 0, 0, 0, 0, self.img_counter)
                        if self.write_img(pix,file_name, figure.alt_colorspace):
                            self.img_counter += 1
                    figure = None

            self.doc.close()
        except KeyboardInterrupt:
            self.doc.close()
            raise
        except :
            self.doc.close()
            print("NORMAL MODE FAILS")
            raise

    def write_alpha_imgs(self, byte_img, name):
        """
        Set white background for RGBA images
        :param byte_img: string with imgs in Byte
        :param name: save filename
        """
        if byte_img:
            img = Image.open(io.BytesIO(byte_img))
            img.load()
            if img.size[0] < 10 or img.size[1] < 10:
                return False
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            background.save(name)
            del background
            del img
            return True
        else:
            raise ValueError("IMAGE %s WITH STENCIL MASK ERROR" % (name))

    def write_img(self, pix, file_name, alt_colorspace=None):
        """
        Write image from pix <fitz.Pixmap> to an file
        :param pix: <fitz.Pixmap>
        :param file_name: <string>
        :param alt_colorspace: Check alernative colorspace.
                                If alt_colorspace is "Separation" or "DeviceN" a color inversion is needed.
        :return: <boolean> success write
        """
        if pix.width < 10 or pix.height < 10:
            return False

        if pix.colorspace:  # This is csGRAY or csRGB
            try:
                if pix.colorspace.name == 'DeviceGray':
                    if pix.alpha == 1:
                        pixRGB = fitz.Pixmap(fitz.csRGB, pix)
                        return  self.write_alpha_imgs(pixRGB.getPNGData(), file_name)
                    else:
                        if alt_colorspace == "Separation" or alt_colorspace == "DeviceN":
                            pix.invertIRect()
                        pix.writePNG(file_name)
                    return True
                elif pix.colorspace.name == 'DeviceRGB':
                    if pix.alpha == 1:
                        return  self.write_alpha_imgs(pix.getPNGData(), file_name)
                    else:
                        pix.writePNG(file_name)
                    return True

                elif pix.colorspace.name == 'DeviceCMYK':  # Only csRGB has writing implementation
                    pixRGB = fitz.Pixmap(fitz.csRGB, pix)  # converting to csRGB
                    if pixRGB.alpha == 1:
                        return self.write_alpha_imgs(pixRGB.getPNGData(), file_name)
                    else:
                        pixRGB.writePNG(file_name)
                    return True
                else:
                    raise ValueError("Error: Unknown image colorspace: %s" % pix.colorspace.name)

            except ValueError:
                return False
            except KeyboardInterrupt:
                raise

        return False

    def build_overlap_set(self, figures):
        """
        Build a list of sets with all figures that has overlap on the page
        :return list of sets with the index of the overlapped images
        """
        overlap_set = [set() for empty_set in range(len(figures))]
        for i in range(len(figures)):
            for j in range(i, len(figures)):
                if self.has_overlap(figures, i, j):
                    overlap_set[i].add(j)

        self.union_intersections_images(overlap_set)

        # Run through overlap_set until it not change
        while True:
            old_len_set = len(overlap_set)
            overlap_set = self.union_region_fig_overlap(figures, overlap_set)
            if old_len_set == len(overlap_set):
                break

        return overlap_set

    def union_region_fig_overlap(self, figures, overlap_set):

        overlap_figs = []

        for index, set_figs in enumerate(overlap_set):
            list_figs = list(set_figs)
            valid_index_bbox = 0
            stop_index = 0
            # Verify the first valid bbox
            for lists_index, figures_index in enumerate(list_figs):
                if figures[figures_index].bbox:
                    valid_index_bbox = lists_index
                    stop_index = lists_index
                    break
            if stop_index <= len(list_figs) - 1 and figures[list_figs[valid_index_bbox]].bbox:
                union_region = figures[list_figs[valid_index_bbox]].copy()
                for fig in list_figs[valid_index_bbox:]:
                    union_region.bbox.includeRect(figures[fig].bbox)
                overlap_figs.append(union_region)
            else:
                overlap_figs.append(ImageEmbedded(None, None))
        # While we find a overlap we restart searching
        restart_union_search = True
        while restart_union_search:
            restart_union_search = False
            for i in range(len(overlap_figs)):
                if restart_union_search:
                    break
                for j in range(i, len(overlap_figs)):
                    if i != j and self.has_overlap(overlap_figs, i, j):
                        if not self.same_location_bbox_used(overlap_set[i], overlap_set[j], figures):
                            overlap_figs[i].bbox.includeRect(overlap_figs[j].bbox)
                            overlap_set[i] = overlap_set[i].union(overlap_set[j])
                            overlap_figs.pop(j)
                            overlap_set.pop(j)
                            restart_union_search = True
                            break

        self.union_intersections_images(overlap_set)

        return overlap_set

    def same_location_bbox_used(self, set_i, set_j, figures):
        """
        Verify if 2 elemens of the set_i and set_j has the same location
        :param set_i with index of the figures:
        :param set_j: with index of the figures
        :param figures: list of ImageEmbedded:
        :return: bool
        """
        for fig_i in set_i:
            for fig_j in set_j:
                p0_i, p1_i, p2_i, p3_i = get_rectangles_points(figures[fig_i].bbox)
                p0_j, p1_j, p2_j, p3_j = get_rectangles_points(figures[fig_j].bbox)
                if p0_i.distance_to(p0_j) == 0 and p1_i.distance_to(p1_j) == 0 \
                        and p2_i.distance_to(p2_j) == 0 and p3_i.distance_to(p3_j) == 0:
                    return True

        return False

    def union_intersections_images(self, overlap_set):
        """
        Union set that has intersections
        """
        restart_union_search = True
        while restart_union_search:
            restart_union_search = False
            for i in range(len(overlap_set)):
                if restart_union_search:
                    break
                for j in range(i, len(overlap_set)):
                    if bool(set.intersection(overlap_set[i], overlap_set[j])) and i != j:
                        overlap_set[i] = overlap_set[i].union(overlap_set[j])
                        del (overlap_set[j])
                        restart_union_search = True
                        break

    def assembly_image(self, figures, file_name):
        """
        Assembly a set of figures that has overlap
        The idea used in this function is merging two image from the overlap set, inserting the result image into the set.
        We do this until one image left, the resulting image.
        Notice: if the overlap set was bad constructed or the image was too damaged, we give up from merging one, and
        try it with the other one.
        :param figures: <list> of overlapped ImageEmbedded objects
        :param file_name: <path name>
        :return: <boolean> success
        """
        # Create a sketch of the resulting image, with the size of the first image of the set
        sketch = fitz.Rect(figures[0].bbox)
        for fig in figures:
            sketch.includeRect(fig.bbox)

        res_img = None
        distance = 1.0  # Initial distance accepted to merge
        try:
            figures.sort(key=lambda x: (x.x1, x.y1, x.x0, x.y0))
            not_found = 1  # flag that showed if has the size of <list> figures, all images from the list are too separete
            # from each other
            while len(figures) > 1:  # until one image is on the set of figures
                obj_i = figures.pop(0)

                for index_j, obj_j in enumerate(figures):
                    if check_overlap(obj_i.bbox, obj_j.bbox, distance):
                        res_img = None
                        res_img, res_obj = self.merge_images(obj_i, obj_j, file_name)
                        figures.pop(index_j)
                        figures.append(res_obj)
                        not_found = 0
                        break

                if not_found == len(figures):  # the set of figures can't find images that are close to each other
                    if distance == 5:
                        distance = 0.5
                        # LOAD IMAGE
                        if obj_i.xref == None:
                            img_i = obj_i.image
                        else:
                            pix = fitz.Pixmap(self.doc, obj_i.xref)
                            if obj_i.colorspace != 3:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            byte_img = pix.getPNGData()
                            img_i = Image.open(io.BytesIO(byte_img))
                            img_i.load()
                            pix = None
                        #Save img_i
                        file_name = file_name + "x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                    (obj_i.x0, obj_i.y0,
                                     obj_i.x1, obj_i.y1, self.img_counter)
                        img_i.save(file_name)
                        self.img_counter += 1
                        continue
                    distance += 0.5
                    print("ERROR: %s IMAGE IS CORRUPT TRYING TO ASSYMBLY WITH MORE %f OF LOST" % (file_name, distance))
                    not_found = 0

                if not_found:  # Reinsert the image_i at the end of the list
                    figures.sort(key=lambda x: (x.x1, x.y1, x.x0, x.y0))
                    figures.append(obj_i)

                not_found += 1

            if res_img != None:  # At this point, the while loop is over and we got the result image
                if res_img.size[0] < 10 or res_img.size[1] < 10:
                    return False
                file_name = file_name + "x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                            (res_obj.x0, res_obj.y0,
                             res_obj.x1, res_obj.y1, self.img_counter)
                res_img.save(file_name)
                return True


            else:
                raise ValueError

        except ValueError:
            print(file_name + " has images errors, on assembly function")
            return False
        except KeyboardInterrupt:
            raise

    def merge_images(self, obj_i, obj_j, file_name):
        """
        Merge two image that has overlapping
        The idea of this method is find the real positions of the two input image, and put them at this position in a
        single image.
        :param obj_i: <ImageEmbedded> object
        :param obj_j: <ImageEmbedded> object
        :param file_name: <string>
        :return: <tuple> result image, result object_embedded
        """
        # LOAD IMAGE
        if obj_i.xref == None:
            img_i = obj_i.image
        else:
            pix = fitz.Pixmap(self.doc, obj_i.xref)
            if obj_i.colorspace != 3 and pix.colorspace and pix.alpha == 0:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            byte_img = pix.getPNGData()
            img_i = Image.open(io.BytesIO(byte_img))
            img_i.load()
            if len(img_i.getbands()) == 1 and pix.colorspace == None:
                img_i = ImageOps.invert(img_i)
                img_i = img_i.convert("RGB")
            if img_i.mode == 'RGBA':
                background = Image.new("RGB", img_i.size, (255, 255, 255))
                background.paste(img_i, mask=img_i.split()[3])
                img_i = background.copy()
                del background


            pix = None
        if obj_j.xref == None:
            img_j = obj_j.image
        else:
            pix = fitz.Pixmap(self.doc, obj_j.xref)
            if obj_j.colorspace != 3 and pix.colorspace and pix.alpha == 0:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            byte_img = pix.getPNGData()
            img_j = Image.open(io.BytesIO(byte_img))
            img_j.load()
            if len(img_j.getbands()) == 1  and pix.colorspace == None :
                img_j = ImageOps.invert(img_j)
                img_j = img_j.convert("RGB")
            if img_j.mode == 'RGBA':
                background = Image.new("RGB", img_j.size, (255, 255, 255))
                background.paste(img_j, mask=img_j.split()[3])
                img_j = background.copy()
                del background

            pix = None

        # Draw a sketch of the resulting image
        sketch = fitz.Rect(obj_i.bbox)
        sketch.includeRect(obj_j.bbox)
        # From the pdf location, get width and height of the resulting image
        real_w = [(sketch.width * obj.width) / (obj.x1 - obj.x0) for obj in [obj_i, obj_j]]
        real_h = [(sketch.height * obj.height) / (obj.y1 - obj.y0) for obj in [obj_i, obj_j]]

        real_w = np.round(np.mean(real_w), 1).astype(int)
        real_h = np.round(np.mean(real_h), 1).astype(int)
        # Check the real offset of each input figure, based on the pdf location and their real sizes
        if obj_i.x0 < obj_j.x0:
            x0_i = 0
            x1_i = real_w * (obj_i.x1 - obj_i.x0) / sketch.width
            x1_j = real_w
            x0_j = real_w - obj_j.width
            if x0_j < 0:
                x0_j = 0
        else:
            x1_i = real_w
            x0_i = real_w - obj_i.width
            if x0_i < 0:
                x0_i = 0
            x0_j = 0
            x1_j = real_w * (obj_j.x1 - obj_j.x0) / sketch.width

        if obj_i.y0 < obj_j.y0:
            y0_i = 0
            y1_i = real_h * (obj_i.y1 - obj_i.y0) / sketch.height
            y1_j = real_h
            y0_j = real_h - obj_j.height
            if y0_j < 0:
                y0_j = 0
        else:
            y1_i = real_h
            y0_i = real_h - obj_i.height
            if y0_i < 0:
                y0_i = 0
            y0_j = 0
            y1_j = real_h * (obj_j.y1 - obj_j.y0) / sketch.height

        x0_i, y0_i, x1_i, y1_i = np.round([x0_i, y0_i, x1_i, y1_i], 3).astype(int)
        x0_j, y0_j, x1_j, y1_j = np.round([x0_j, y0_j, x1_j, y1_j], 3).astype(int)

        # Check if there is any overlapping on the resulting image. If so, write the input image in different images
        if x0_i < x0_j:
            if (x0_i + img_i.width - x0_j) > 10:  # Overlapping > 10
                if (img_j.width > 10 and img_j.height > 10):
                    file_name = file_name + "x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                (obj_j.x0, obj_j.y0,
                                 obj_j.x1, obj_j.y1, self.img_counter)
                    img_j.save(file_name)
                    self.img_counter += 1
                return img_i, obj_i
        if x0_j < x0_i:
            if (x0_j + img_j.width - x0_i) > 10:  # Overlapping > 10
                if (img_i.width > 10 and img_i.height > 10):
                    file_name = file_name + "x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                (obj_i.x0, obj_i.y0,
                                 obj_i.x1, obj_i.y1, self.img_counter)
                    img_i.save(file_name)
                    self.img_counter += 1
                return img_j, obj_j
        if y0_i < y0_j:
            if (y0_i + img_i.height - y0_j) > 10:  # Overlapping > 10
                if (img_j.width > 10 and img_j.height > 10):
                    file_name = file_name + "x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                (obj_j.x0, obj_j.y0,
                                 obj_j.x1, obj_j.y1, self.img_counter)
                    img_j.save(file_name)
                    self.img_counter += 1
                return img_i, obj_i
        if y0_j < y0_i:
            if (y0_j + img_j.height - y0_i) > 10:  # Overlapping > 10
                if (img_i.width > 10 and img_i.height > 10):
                    file_name = file_name + "x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                                (obj_i.x0, obj_i.y0,
                                 obj_i.x1, obj_i.y1, self.img_counter)
                    img_i.save(file_name)
                    self.img_counter += 1
                return img_j, obj_j

        # Create the resulting image with a white background
        res_img = Image.new('RGB', (real_w, real_h), (255, 255, 255))

        res_img.paste(img_j, (x0_j, y0_j))
        res_img.paste(img_i, (x0_i, y0_i))
        # Resulting object
        merge_embbed_image = ImageEmbedded(None, sketch)
        merge_embbed_image.height = real_h
        merge_embbed_image.width = real_w
        merge_embbed_image.colorspace = obj_j.colorspace
        merge_embbed_image.ext = obj_j.ext
        merge_embbed_image.image = res_img

        return res_img, merge_embbed_image

    def has_overlap(self, figures, figure_i, figure_j):
        """
        Check if figure_i and figure_j has overlap in PDF
         Params:
            - figures: <list of ImageEmbedded> of all ImageEmbedded objects on a page
            - figure_i: <int> index i of object i  on the figures list
            - figure_j: <int> index j of object j on the figures list
        return: <bool>
        """
        if figure_i == figure_j:
            return True
        img_i = figures[figure_i]
        img_j = figures[figure_j]
        if img_i.bbox is None or img_j.bbox is None:
            return False
        if img_i.has_alpha() or img_j.has_alpha():
            return False
        if img_i.xref and img_j.xref:
            if img_i.filter != img_j.filter:
                return False

        if img_i.colorspace != img_j.colorspace:
            return False
        if img_i.alt_colorspace != img_j.alt_colorspace:
            return False

        if not (img_j.xref is None):
            if not (img_i.xref is None):
                pixi = fitz.Pixmap(self.doc, img_i.xref)
                pixj = fitz.Pixmap(self.doc, img_j.xref)
                if pixi.colorspace != pixj.colorspace:
                    return False

        return check_overlap(img_i.bbox, img_j.bbox)


def main():
    extract = PDFExtractor()
    extract.extract_all(mode="safe")


if __name__ == "__main__":
    main()
