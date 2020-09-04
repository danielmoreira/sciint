import string
import argparse

from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine

from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk

# Configuration
LANGUAGE = 'english'
IMAGE_CAPTION_LABELS = ['figure', 'fig.']
MAX_WORD_CONCAT_TRIALS = 128
NON_PUNCTUATION_TABLE = {ord(char): None for char in string.punctuation}

# Shared variables.
# other variables should be added here...
PAPER_VOCABULARY = []  # holds the current paper's vocabulary


# Removes the punctuation from a given string.
def _remove_punctuation(text):
    # removes any latin alphabet punctuation
    output = text.translate(NON_PUNCTUATION_TABLE)

    # removes english possession punctuation ('s)
    if text[-2:] == "'s":
        output = text[:-2].translate(NON_PUNCTUATION_TABLE)

    # TODO: add further punctuation cases here

    # output
    return output


# Removes unnecessary white spaces from the given string (a common problem from pdf-miner extraction).
def _remove_unnecessary_white_spaces(text):
    output = []

    i = 0
    content = text.split()  # splits the given text into char chunks

    while i < len(content):
        last_valid_word_position = i
        word_acc = content[i].strip()

        for j in range(i + 1, min(MAX_WORD_CONCAT_TRIALS, len(content))):
            # tries to concatenate next char chunk
            word_acc = word_acc + content[j].strip()

            # removes punctuation
            non_punctuated_word = _remove_punctuation(word_acc)

            # if the current concatenation is a valid word, stores the last valid position
            if wordnet.synsets(non_punctuated_word) or \
                    non_punctuated_word in stopwords.words(LANGUAGE) or \
                    non_punctuated_word in PAPER_VOCABULARY:
                last_valid_word_position = j

        # obtains the last valid word, based on the just computed last valid position
        word = ''
        start = i
        for j in range(start, last_valid_word_position + 1):
            word = word + content[j].strip()
            i = i + 1

        # adds a non punctuated version of the last valid word to the paper vocabulary
        non_punctuated_word = _remove_punctuation(word)
        if non_punctuated_word not in PAPER_VOCABULARY:
            PAPER_VOCABULARY.append(non_punctuated_word)

        # adds the last valid word to the method output
        output.append(word + ' ')

    return ''.join(output).strip()


# Extracts text from the PDF file stored in the give path.
# The text is stored in a dictionary, whose keys are the PDF page numbers.
def _extract_pdf_text(pdfFilePath):
    output = {}  # stores the obtained text; content is grouped by page number

    with open(pdfFilePath, 'rb') as file:
        # pdf parser and pdf document object
        pdf_parser = PDFParser(file)
        pdf_doc = PDFDocument()
        pdf_parser.set_document(pdf_doc)
        pdf_doc.set_parser(pdf_parser)

        pdf_doc.initialize('')

        # pdf resource manager, page aggregator, and page interpreter
        pdf_resource_manager = PDFResourceManager()
        pdf_page_aggregator = PDFPageAggregator(pdf_resource_manager, laparams=LAParams())
        pdf_page_interpreter = PDFPageInterpreter(pdf_resource_manager, pdf_page_aggregator)

        # for each obtained page
        for page in pdf_doc.get_pages():
            key = pdf_page_aggregator.pageno
            output[key] = []

            pdf_page_interpreter.process_page(page)
            pdf_content = pdf_page_aggregator.get_result()

            page_height = page.mediabox[3]

            for obj in pdf_content:
                if isinstance(obj, LTTextBox) or isinstance(obj, LTTextLine):
                    paragraph = _remove_unnecessary_white_spaces(obj.get_text()) + '\n'

                    x0 = obj.bbox[0]
                    y0 = page_height - obj.bbox[3]
                    x1 = obj.bbox[2]
                    y1 = page_height - obj.bbox[1]

                    output[key].append([paragraph.strip(), (x0, y0, x1, y1)])

    return output


# Extracts the figure captions from the given PDF text content.
# The figure captions are returned in a list, in the order they happen inside the PDF file.
# Each element of the list is a triple (caption text, page number, bounding box).
def _extract_figure_captions(pdf_text):
    output = []

    # for each page of the PDF
    for page in pdf_text:
        # for each paragraph of the current page
        for paragraph in pdf_text[page]:
            # gets the sentences of the current paragraph
            sentences = nltk.sent_tokenize(paragraph[0].strip(), LANGUAGE)

            # if there are sentences...
            if len(sentences) > 0:
                # gets the words of the 1st sentence
                first_sentence_words = sentences[0].split()

                # if the 1st word is a caption stop word
                if len(first_sentence_words) > 0 and first_sentence_words[0].lower() in IMAGE_CAPTION_LABELS:
                    output.append([paragraph[0].strip(), page, paragraph[1]])

    return output


# Extracts the figure paths from the given PDF file.
# Returns a list with the obtained figure captions, in the order they happen inside the PDF file.
# Each element of the list is a triple [caption text, page number, bounding box].
def extract_figure_captions(pdf_file_path):
    pdf_text = _extract_pdf_text(pdf_file_path)
    captions = _extract_figure_captions(pdf_text)
    return captions


# Usage example
def usage_example():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf_path')
    args = parser.parse_args()

    captions = extract_figure_captions(args.pdf_path)
    for caption in captions:
        print(caption)


if __name__ == "__main__":
    usage_example()
