import sys
import os
import math
from .caption_extractor import extract_figure_captions as extract_caption
import argparse


# Parses a given figure file name, to obtain its source page number and bounding box.
# File name example: p-2-x0-242.411-y0-368.515-x1-287.105-y1-466.669-5.png.
# Expected output: (2, 242.411, 368.515, 287.105, 466.669)
# Meaning: page 2, bounding box ((242.411, 368.515), (287.105, 466.669)).
def _parse_figure_file_name(figure_file_name):
    try:
        content = figure_file_name.replace('--', '-*')
        content = content.split('-')
        page = int(content[1])
        x0 = float(content[3].replace('*', '-'))
        y0 = float(content[5].replace('*', '-'))
        x1 = float(content[7].replace('*', '-'))
        y1 = float(content[9].replace('*', '-'))

    except:
        raise Exception('[ERROR] Could not parse file name', figure_file_name + '.')

    return figure_file_name, page, (x0, y0, x1, y1)


# Greedy algorithm that tries to link captions to figures, based on their page positions.
# Returns the list of captions with the same size of <figure_positions>,
# since it attributes one caption to each figure.
def _link_captions_to_figures(figure_positions, caption_positions, testing=False):
    output = []

    # for each figure
    for figure_position in figure_positions:
        # retrieves only the captions whose page is the same of the figure
        candidate_caption_positions = []
        for caption_position in caption_positions:
            if caption_position[1] == figure_position[1]:  # the page number is the same
                candidate_caption_positions.append(caption_position)

        if len(candidate_caption_positions) == 0:
            if testing:
                output.append(('', -1, (-1,-1,-1,-1)))
            else:
                raise Exception('[ERROR] There are no captions on page', figure_position[1], 'where figure',
                            figure_position[0], 'is depicted.')

        elif len(candidate_caption_positions) == 1:
            output.append(candidate_caption_positions[0])

        else:
            # tries to find the caption that bounds the current figure,
            # or the closest caption with respect to top-left corners
            found_caption = False
            closest_caption_position = None
            closest_distance = sys.maxsize

            for caption_position in candidate_caption_positions:
                if figure_position[2][0] >= caption_position[2][0] and \
                        figure_position[2][2] <= caption_position[2][2] and \
                        figure_position[2][1] >= caption_position[2][1] and \
                        figure_position[2][3] <= caption_position[2][3]:
                    found_caption = True
                    output.append(caption_position)
                    break

                else:
                    current_distance = math.sqrt((figure_position[2][0] - caption_position[2][0]) ** 2 +
                                                 (figure_position[2][1] - caption_position[2][1]) ** 2)
                    if current_distance < closest_distance:
                        closest_caption_position = caption_position
                        closest_distance = current_distance

            # if it did not find the best caption yet
            if not found_caption:
                # sets the caption as the closest one and updates its boundaries
                output.append(closest_caption_position)
                closest_caption_position[2] = (min(figure_position[2][0], closest_caption_position[2][0]),
                                               min(figure_position[2][1], closest_caption_position[2][1]),
                                               max(figure_position[2][2], closest_caption_position[2][2]),
                                               max(figure_position[2][3], closest_caption_position[2][3]))

    # returns the list of captions for each given figure
    return output


# Extracts captions and attributes them to the give figure file paths.
# Returns a list of (figure-file-path, caption-text) pairs.
def extract_figure_captions(figure_list_file_path, pdf_file_path):
    # obtains the list of figure file paths
    figure_file_paths = []
    with open(figure_list_file_path) as f:
        for line in f:
            figure_file_paths.append(line.strip())

    # parses the list of figure files paths, to obtain their page and positions
    figure_positions = []
    for fp in figure_file_paths:
        figure_positions.append(_parse_figure_file_name(fp.split(os.path.sep)[-1]))

    # obtains the captions from the given pdf file
    caption_positions = extract_caption(pdf_file_path)

    # greedly links captions to figures
    caption_list = _link_captions_to_figures(figure_positions, caption_positions)

    # returns a list with images and their respective captions
    output = []
    for i in range(len(figure_file_paths)):
        output.append((figure_file_paths[i], caption_list[i][0]))
    return output


# Usage example
def usage_example():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_list_path')
    parser.add_argument('pdf_path')
    args = parser.parse_args()

    captions = extract_figure_captions(args.image_list_path, args.pdf_path)
    for caption in captions:
        print(caption)


if __name__ == "__main__":
    usage_example()
