"""
This program segments the images of PDF pages and extracts the figures with image processing techniques.
Author: Daniel Moreira (July 2020)
"""

import os
import io
import shutil
import argparse
import numpy
import cv2
import pytesseract
import fitz
from PIL import Image

# Configuration values.
# Image patch size for analysis, in pixels. Size x size patches are analysed and used as white separating borders.
PATCH_SIZE = 15

# Patch analysis offset, in pixels.
PATCH_OFFSET = 5

# Remaining content at-least threshold ([0-1] rate) to consider a figure useful, after removing text.
FIGURE_TRSH = 0.1

# Extracted PDF page image width, in pixels
PAGE_WIDTH = 1224


# Tests if a given gray-scaled patch image is empty (i.e., mostly white).
# Returns TRUE if it is empty, FALSE otherwise.
def _is_empty(gs_patch):
    return numpy.mean(gs_patch) >= 250


# Removes, inplace, all the text within a given gray-scaled image.
def _remove_text(gs_image):
    h, w = gs_image.shape

    # removes all the text within the given image
    text = pytesseract.image_to_boxes(gs_image)
    text = [letter.split() for letter in text.split('\n')]
    for _, letter in enumerate(text):
        if len(letter) >= 5:
            r0 = int(letter[1])
            c0 = h - int(letter[2])
            r1 = int(letter[3])
            c1 = h - int(letter[4])

            if letter[0] not in ['~']:  # to avoid the edges of wire-frame figures
                cv2.rectangle(gs_image, (r0, c0), (r1, c1), (255, 255, 255), 5)
                cv2.rectangle(gs_image, (r0, c0), (r1, c1), (255, 255, 255), -1)


# Verifies if a given (row, col) position lies within any of the given rectangles.
# Returns TRUE if it lies within, FALSE otherwise.
def _lies_within_rectangle(rectangles, row, col):
    for r in rectangles:
        if row >= r[0] and row < r[2] and col >= r[1] and col < r[3]:
            return True

    return False


# Verifies if a reference rectangle (row_0, col_0, row_1, col_1) intersects any of the given rectangles.
# Returns TRUE if it intersects, FALSE otherwise.
def _intersects_rectangle(rectangles, r0, c0, r1, c1):
    # for each given rectangle...
    for r in rectangles:
        # any reference rectangle's corner within the current rectangle?
        if ((r0 >= r[0] and r0 < r[2]) or (r1 > r[0] and r1 <= r[2])) and (
                (c0 >= r[1] and c0 < r[3]) or (c1 > r[1] and c1 <= r[3])):
            return True

        # any current rectangle's corner within the reference one?
        if ((r[0] >= r0 and r[0] < r1) or (r[2] > r0 and r[2] <= r1)) and (
                (r[1] >= c0 and r[1] < c1) or (r[3] > c0 and r[3] <= c1)):
            return True

        # does the reference rectangle lie across the current one (column-wise)?
        if r0 < r[0] and r1 > r[2] and c0 > r[1] and c0 < r[3]:
            return True

        # does the reference rectangle lie across the current one (row-wise)?
        if r0 > r[0] and r0 < r[2] and c0 < r[1] and c1 > r[3]:
            return True

    # default answer: FALSE
    return False


# Divides the given gray-scaled image into column-wise partitions separated
# by white spaces of size PATCH_SIZE x PATCH_SIZE pixels on either vertical and horizontal directions.
# Returns the list of obtained column-wise rectangles.
def _column_wise_partition(gs_image):
    # holds the obtained partitions
    partitions = []

    # image dimensions
    h, w = gs_image.shape

    # current partition references
    r0 = h
    c0 = w
    r1 = 0
    c1 = 0

    # column-wise image patch analysis
    col = 0
    while col < w:
        # row-wise image patch analysis
        row = 0
        while row < h:
            # if the current position is not already within the current partition, processes it
            if not _lies_within_rectangle([(r0, c0, r1, c1)], row + PATCH_SIZE, col + PATCH_SIZE):
                # is the current position within any previously processed partition?
                is_within_partition = _lies_within_rectangle(partitions, row, col)

                # if the current position is out of a partition, processes it
                if not is_within_partition:
                    # current patch
                    patch = gs_image[row:min(row + PATCH_SIZE, h), col:min(col + PATCH_SIZE, w)]

                    # if there is not a partition being detected yet
                    if r0 > r1 or c0 > c1:
                        # non-empty patch
                        if not _is_empty(patch):
                            # starts a new segment
                            r0 = row
                            c0 = col
                            r1 = min(row + patch.shape[0], h)
                            c1 = min(col + patch.shape[1], w)

                            # # TODO remove for visualization debug
                            # cp = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)
                            # cv2.rectangle(cp, (c0, r0), (c1 - 1, r1 - 1), (0, 0, 255), 1)
                            # cv2.rectangle(cp, (col, row), (col + patch.shape[1], row + patch.shape[0]), (0, 255, 0), -1)
                            # cv2.imshow('page', cp)
                            # cv2.waitKey(30)

                        # else, empty patch, so nothing needs to be done

                    # else, we have a partition being detected
                    else:
                        # non-empty patch
                        if not _is_empty(patch):
                            # probable new reference partition
                            new_r0 = min(r0, row)
                            new_c0 = min(c0, col)
                            new_r1 = max(r1, min(row + patch.shape[0], h))
                            new_c1 = max(c1, min(col + patch.shape[1], w))

                            # if probable new reference partition is not intersecting a previous one, processes it
                            if not _intersects_rectangle(partitions, new_r0, new_c0, new_r1, new_c1):
                                # if r1 has changes, reviews rows and columns of current segment
                                if r1 != new_r1:
                                    row = new_r0  # - PATCH_OFFSET
                                    col = new_c0  # - PATCH_OFFSET

                                # makes the new partition as the current one (updates the current partition)
                                r0 = new_r0
                                c0 = new_c0
                                r1 = new_r1
                                c1 = new_c1

                                # # TODO remove for visualization debug
                                # cp = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)
                                # cv2.rectangle(cp, (c0, r0), (c1 - 1, r1 - 1), (0, 0, 255), 1)
                                # cv2.rectangle(cp, (col, row), (col + patch.shape[1], row + patch.shape[0]), (0, 255, 0),
                                #               -1)
                                # cv2.imshow('page', cp)
                                # cv2.waitKey(30)

                        # else, empty patch
                        else:
                            # if the current patch surpasses the bottom of the current partition
                            if row >= r1 or row + PATCH_OFFSET >= h:
                                # restarts row count and moves columns forward
                                row = 0
                                col = col + PATCH_OFFSET
                                while _lies_within_rectangle(partitions, row, col):
                                    row = row + PATCH_OFFSET

                                # if current patch is on the right of the current partition (in addition to the bottom)
                                if col >= c1 or col + PATCH_OFFSET >= w:
                                    # restarts column count
                                    row = 0
                                    col = 0
                                    while _lies_within_rectangle(partitions, row, col):
                                        row = row + PATCH_OFFSET

                                    # finalizes and adds the current partition to the list of partitions
                                    if r1 > r0 and c1 > c0:
                                        partitions.append((r0, c0, r1, c1))

                                        # resets the current partition references
                                        r0 = h
                                        c0 = w
                                        r1 = 0
                                        c1 = 0

                                # skips next row count update
                                continue

            row = row + PATCH_OFFSET
        col = col + PATCH_OFFSET

    # adds the remaining partition to the list of partitions
    if r1 > r0 and c1 > c0:
        partitions.append((r0, c0, r1, c1))

    return partitions


# Divides the given gray-scaled image into row-wise partitions separated
# by white spaces of size PATCH_SIZE x PATCH_SIZE pixels on horizontal direction only.
# Returns the list of obtained row-wise rectangles.
def _row_wise_partition(gs_image):
    # holds the obtained partitions
    partitions = []

    # image dimensions
    h, w = gs_image.shape

    # current partition references
    r0 = h
    c0 = w
    r1 = 0
    c1 = 0

    # row-wise image patch analysis
    row = 0
    while row < h:
        # column-wise image patch analysis
        col = 0
        while col < w:
            # is the current position within any partition?
            is_within_partition = _lies_within_rectangle(partitions, row, col)

            # if the current position is out of a partition, processes it
            if not is_within_partition:
                # current image patch
                patch = gs_image[row:min(row + PATCH_SIZE, h), col:min(col + PATCH_SIZE, w)]

                # if there is not a partition being detected yet
                if r0 > r1 or c0 > c1:
                    # non-empty patch
                    if not _is_empty(patch):
                        # starts a new partition
                        r0 = row
                        c0 = col
                        r1 = min(row + patch.shape[0], h)
                        c1 = min(col + patch.shape[1], w)

                        # # TODO remove for visualization debug
                        # cp = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)
                        # cv2.rectangle(cp, (c0, r0), (c1 - 1, r1 - 1), (0, 0, 255), 1)
                        # cv2.rectangle(cp, (col, row), (col + patch.shape[1], row + patch.shape[0]), (0, 255, 0), -1)
                        # cv2.imshow('page', cp)
                        # cv2.waitKey(30)

                    # else, empty patch, so nothing needs to be done

                # else, we have a partition being detected
                else:
                    # non-empty patch
                    if not _is_empty(patch):
                        # updates the reference partition
                        r0 = min(r0, row)
                        c0 = min(c0, col)
                        r1 = max(r1, min(row + patch.shape[0], h))
                        c1 = max(c1, min(col + patch.shape[1], w))

                        # # TODO remove for visualization debug
                        # cp = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)
                        # cv2.rectangle(cp, (c0, r0), (c1 - 1, r1 - 1), (0, 0, 255), 1)
                        # cv2.rectangle(cp, (col, row), (col + patch.shape[1], row + patch.shape[0]), (0, 255, 0), -1)
                        # cv2.imshow('page', cp)
                        # cv2.waitKey(30)

                    # else, empty patch
                    else:
                        # if current patch is the rightmost one
                        if col + PATCH_OFFSET >= w:
                            # restarts column count and moves row forward
                            col = 0
                            row = row + PATCH_OFFSET
                            while _lies_within_rectangle(partitions, row, col):
                                col = col + PATCH_OFFSET

                            # if current patch is on the bottom of the current partition (in addition to the right)
                            if row >= r1 or row + PATCH_OFFSET >= h:
                                # restarts row count
                                row = 0
                                col = 0
                                while _lies_within_rectangle(partitions, row, col):
                                    col = col + PATCH_OFFSET

                                # adds the current partition to list of partitions
                                if r1 > r0 and c1 > c0:
                                    partitions.append((r0, c0, r1, c1))

                                    # resets the current partition references
                                    r0 = h
                                    c0 = w
                                    r1 = 0
                                    c1 = 0

                            # skips next column count update
                            continue

            col = col + PATCH_OFFSET
        row = row + PATCH_OFFSET

    # adds the remaining partition to list of partitions
    if r1 > r0 and c1 > c0:
        partitions.append((r0, c0, r1, c1))

    return partitions


# Evaluates the given gray-scale image and crops it keeping only figure information.
# Returns the figure bounding box (row_0, col_0, row_1, col_1) or None if there's no figure on the given image.
def _filter_figure_in(gs_image):
    # image dimensions
    h, w = gs_image.shape

    # removes all the text within the given image
    _remove_text(gs_image)

    # finds contours on the remaining content
    # _, contours, _ = cv2.findContours(cv2.Canny(gs_image, 1, 2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(cv2.Canny(gs_image, 1, 2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        # current bounding box
        r0 = h
        c0 = w
        r1 = 0
        c1 = 0

        # merges all large-enough contours into one (large enough: more than 9 pixels)
        for c in contours:
            _x, _y, _w, _h = cv2.boundingRect(c)
            if _w * _h > 9:
                if _y < r0:
                    r0 = _y

                if _x < c0:
                    c0 = _x

                if _y + _h > r1:
                    r1 = _y + _h

                if _x + _w > c1:
                    c1 = _x + _w

        # if the obtained bounding box has at least FIGURE_TRSH times the area of the given image, returns it
        if r1 > r0 and c1 > c0 and (r1 - r0) * (c1 - c0) / float(h * w) >= FIGURE_TRSH:
            return r0, c0, r1, c1

    # default answer: no figure
    return None


# Extracts the pages of the given PDF file as images and stores them in the given output folder.
# Returns a list with the absolute file paths of the extracted images, as well as each page original size.
def _extract_images_page_print(pdf_file_path, output_folder):
    output = []
    sizes = []

    doc = fitz.open(pdf_file_path)
    for page_number, page in enumerate(doc):
        page_number = page_number + 1

        page_image = numpy.array(
            Image.open(io.BytesIO(page.getPixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False).getImageData())))

        h, w = page_image.shape[0:2]
        new_w = PAGE_WIDTH
        new_h = int(round(PAGE_WIDTH * h / float(w)))

        page_image = cv2.resize(page_image, (new_w, new_h))

        filename = output_folder + '/' + f"page-{page_number:03d}.png"
        cv2.imwrite(filename, page_image)
        output.append(filename)

        sizes.append((h / 2.0, w / 2.0))

    return output, sizes


# Main method.
# Takes a pdf file, extracts its pages as images, and segments them into figures.
def extract_figures(pdf_file_path, output_folder):
    # throws an exception if output folder does not exist
    if not os.path.isdir(output_folder):
        raise Exception('Output folder ' + output_folder + ' does not exist.')

    # prepares the output folder
    output_folder = output_folder + '/' + pdf_file_path.split('/')[-1].split('.')[0]
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    # extracts all pages from the given PDF as images
    page_images, page_sizes = _extract_images_page_print(pdf_file_path, output_folder)

    # for each page
    image_count = 0
    for p in range(len(page_images)):
        # reads the page and its gray-scale version
        page = cv2.imread(page_images[p])
        gs_page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

        # performs a row-wise page segmentation
        rectangles_1 = _row_wise_partition(gs_page)

        # performs a column-wise page segmentation
        rectangles_2 = []
        for r in rectangles_1:
            gs_part = gs_page[r[0]:r[2], r[1]:r[3]]
            sub_rectangles = _column_wise_partition(gs_part)
            for s in sub_rectangles:
                rectangles_2.append((r[0] + s[0], r[1] + s[1], r[0] + s[2], r[1] + s[3]))

        # performs a row-wise page segmentation, again
        rectangles_3 = []
        for r in rectangles_2:
            gs_part = gs_page[r[0]:r[2], r[1]:r[3]]
            sub_rectangles = _row_wise_partition(gs_part)
            for s in sub_rectangles:
                rectangles_3.append((r[0] + s[0], r[1] + s[1], r[0] + s[2], r[1] + s[3]))

        # filters out undesired figures
        rectangles_4 = []
        parts_4 = []
        for r in rectangles_3:
            gs_part = gs_page[r[0]:r[2], r[1]:r[3]]
            part = _filter_figure_in(gs_part)
            if part is not None:
                part = (r[0] + part[0], r[1] + part[1], r[0] + part[2], r[1] + part[3])
                rectangles_4.append(part)
                parts_4.append(page[part[0]:part[2], part[1]:part[3]])

        # saves the obtained images
        for r in range(len(rectangles_4)):
            image_count = image_count + 1
            size_ratio = page_sizes[p][1] / PAGE_WIDTH
            filename = "%s/p-%s-x0-%.3f-y0-%.3f-x1-%.3f-y1-%.3f-%d.png" % \
                       (output_folder, str(p + 1),
                        rectangles_4[r][1] * size_ratio, rectangles_4[r][0] * size_ratio,
                        rectangles_4[r][3] * size_ratio, rectangles_4[r][2] * size_ratio, image_count)
            cv2.imwrite(filename, parts_4[r])

        # removes the extracted page image
        os.remove(page_images[p])

# # Makes this file executable.
# parser = argparse.ArgumentParser(prog='page_segmenter.py',
#                                  description="This program segments the images of each PDF page to extract figures.")
# parser.add_argument("pdfPath", metavar=('PATH_TO_PDF'),
#                     help="The path to the PDF whose images must be extracted.")
# parser.add_argument("resPath", metavar=('PATH_TO_RESULT'),
#                     help="The path to the result folder.")
# args = parser.parse_args()
#
#
# def main():
#     extract_figures(args.pdfPath, args.resPath)
#
#
# if __name__ == "__main__":
#     main()
