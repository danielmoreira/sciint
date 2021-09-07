#!/usr/bin/env bash
# Parameters:

# $1: Path to the query image.

# $2: Path to the file listing the images of the dataset.

# $3: Path to the output file containing the image rank.

python3.7 -u ./src/flat_query.py $1 $2 $3
