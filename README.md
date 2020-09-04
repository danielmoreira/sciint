# PDF Content Extraction 
In this branch you find the code and experimental results related with Image and Caption extraction.

## Image Extraction

### Instruction

A Dockerfile wrapping all needed environment is set in the `/docker`. 

To build the docker use:

​	`$ cd docker && ./build`

After this, use `run_image_extraction.sh` file to communicate with the docker and perform image extraction.

` $ run_image_extraction.sh <pdf_path> <output_path> ` 

The experimental setup and result for the image extraction module are summarized in the `PDF Image Extraction` notebook.

