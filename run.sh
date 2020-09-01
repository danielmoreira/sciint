#!/bin/bash
function help
{
    echo "Usage: run.sh INPUT_PDF_PATH OUTPUT_PATH"
    echo 
    echo "    INPUT_PDF_PATH must be a file"
    echo "    OUTPUT_PATH must be a directory"
    echo
    echo 'Example:'
    echo '$  run.sh sample.pdf output/'
    exit 1
}

function isdir
{
    [ -d "$1" ]
}

function isfile
{
    [ -f "$1" ]
}

function valid_arguments
{
     if ! isfile "$1"
         then 
             echo "'$1'" is not a valid INPUT FILE
             help
     fi
     if ! isdir "$2" 
         then
             echo "'$2'" is not a valid OUTPUT FOLDER
             help
     fi
}

# Check number of arguments
if ! [[ $# -eq 2 ]]
then
    help

else
    INPUT_PDF_PATH=$1
    OUTPUT_PATH=$(realpath $2)

    # Valid Args
    valid_arguments "$INPUT_PDF_PATH" "$OUTPUT_PATH"  

    # Get directory of input_pdf
     INPUT_DIR=$(realpath $(dirname "$INPUT_PDF_PATH"))
     INPUT_BASENAME=$(basename "$INPUT_PDF_PATH")

    # Run docker
    docker run --userns=host  --rm \
    --user "$(id -u):$(id -g)" \
    -e PDF="$INPUT_BASENAME" \
    -v "$INPUT_DIR":/INPUT \
    -v "$OUTPUT_PATH":/OUTPUT \
    pdf-content-extraction

fi
