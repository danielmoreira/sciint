import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../PDF-Image-Extraction/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../print_page/')
import page_segmenter
from pdfExtractor import PDFExtractor


# Process the program input arguments.
# Returns the given PDF file path and output folder path.
def _read_input():
    parser = argparse.ArgumentParser(prog='pdf_image_extractor.py', description="Image extraction from PDF files.",
                                     formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument("--pdf_path", "-i", required=True, type=str, nargs=1,
                        help="Path to the PDF file whose image must be extracted.")
    parser.add_argument("--output_folder", "-o", required=True, type=str, nargs=1,
                        help="Output folder, where extracted images will be saved.")
    args = vars(parser.parse_args())

    pdf_file_path = args['pdf_path'][0]
    output_folder = args['output_folder'][0]

    return pdf_file_path, output_folder


# Extracts the images from the given PDF file with Unicamp's method, and stores them in the given output folder.
# Returns a list with the absolute file paths of the extracted images.
def _extract_images_unicamp(pdf_file_path, output_folder):
    id = pdf_file_path.split('/')[-1].split('.')[0]

    image_extractor = PDFExtractor(input_path=pdf_file_path)
    image_extractor.extract_all(out_name=output_folder)

    image_list = []
    for file in os.listdir(output_folder + '/' + id):
        if file.split('.')[-1] in ['png']:
            image_list.append(os.path.abspath(output_folder) + '/' + id + '/' + file)

    return image_list


# Extracts the images from the given PDF file by generating images of each PDF page
# and segmenting them with image processing operations.
def _extract_images_page_segmenter(pdf_file_path, output_folder):
    id = pdf_file_path.split('/')[-1].split('.')[0]

    page_segmenter.extract_figures(pdf_file_path, output_folder)

    image_list = []
    for file in os.listdir(output_folder + '/' + id):
        if file.split('.')[-1] in ['png']:
            image_list.append(os.path.abspath(output_folder) + '/' + id + '/' + file)

    return image_list


# Tests if Unicamp's image extraction method has failed
# (i.e., it generated one image for each PDF page)
def _unicamp_fail(image_file_paths):
    # recipe: if there is one image per page and all of them start at (0, 0)
    pages = []
    for p in image_file_paths:
        p = p.replace('--', '-*')
        parts = p.split('/')[-1].split('-')

        x0 = float(parts[3].replace('*', '-'))
        y0 = float(parts[5].replace('*', '-'))
        if x0 > 0.0 or y0 > 0.0:
            return False

        page = int(parts[1])
        if page in pages:
            return False
        else:
            pages.append(page)

    if len(pages) == len(image_file_paths):
        return True

    # default: no fail
    return False


# Main method.
def main():
    # program input
    pdf_file_path, output_folder = _read_input()

    # creates the output folder if necessary
    output_folder = output_folder + "/xfigs/"
    os.makedirs(output_folder,exist_ok=True)
    # extracts images with Unicamp's method
    unicamp_images = _extract_images_unicamp(pdf_file_path, output_folder)

    # if Unicamp's method has failed
    if _unicamp_fail(unicamp_images):
        # removes the obtained Unicamp's images
        for i in unicamp_images:
            os.remove(i)

        # extracts images with the page image segmenter
        pageseg_images = _extract_images_page_segmenter(pdf_file_path, output_folder)


if __name__ == "__main__":
    main()
