# Copy Move Detection

Source code and test script of [SILA](https://github.com/danielmoreira/sciint/tree/master) copy-move detection.

## Installation

1. Install [Git.](https://github.com/git-guides/install-git)
2. Install [Git Large File Storage.](https://git-lfs.github.com/)
3. Check the project out from GitLab:
    ```
    git lfs clone --branch copy-move-detection https://github.com/danielmoreira/sciint.git sci-cmfd
    cd sci-cmfd
    ```

#### Installation with Container

4. Install [Docker.](https://docs.docker.com/get-docker/)
5. Build the Docker container. In a terminal, execute:
    ```
    docker build . -t sci-cmfd:latest
    ```

#### Installation without Container

4. Install Python 3.7.7 with PyTorch 1.6.0 and Torchvision 0.7.0
5. Install python requirements:
    ```
    cd ./src
    pip install -r ./requirements.txt
    bash ./download_CRAFT.sh
    cd ..
    ```

## Data Extraction 

1. Create an IO folder:
    ```
    export CMFD_IO=~/CMFD_IO; mkdir -p $CMFD_IO
    ```
   Please change the location of the folder within your machine accordingly.
   
2. Unzip test data in the IO folder:
    ```
    unzip ./copy-move-data/gt.zip -d $CMFD_IO
    unzip ./copy-move-data/figures.zip -d $CMFD_IO
    ```
   Contact [Daniel Moreira](daniel.moreira@nd.edu) to get the password.

## Test Execution 

To execute the test on the 180 images with container. In a terminal, execute:
 ```
 docker run -v $CMFD_IO:/cmfd/io sci-cmfd:latest ./scripts/01_cmfd_all_images.py
 ```

To execute the test without container, execute:
  ```
  python ./scripts/01_cmfd_all_images.py
  ```
    
## Metric Collection

We use the F1 ([F-measure](https://en.wikipedia.org/wiki/F-score)) metric to evaluate the copy-move detection.
to compute F1 metric with container, execute:
 ```
 docker run -v $CMFD_IO:/cmfd/io sci-cmfd:latest ./scripts/02_eval_all_results.py
 ```

To compute F1 metric without container, execute:
 ```
 python ./scripts/02_eval_all_results.py
 ```   

The result is:
 ```
 Number of images: 180
 ****************************************
 |        |   RGB   | Zernike |  Merge  |
 |F1 mean |  0.325  |  0.307  |  0.345  |
 |F1 std  |  0.31   |  0.31   |  0.30   |
 ****************************************
 ```
 
 
 ## Cite this Work
Please cite as:
> Moreira, D., Cardenuto, J.P., Shao, R. et al. SILA: a system for scientific image analysis. Nature Scientific Reports 12 (18306), 2022.
> https://doi.org/10.1038/s41598-022-21535-3

```
@article{sila,
   author = {Moreira, Daniel and Cardenuto, João Phillipe and Shao, Ruiting and Baireddy, Sriram and Cozzolino, Davide and Gragnaniello, Diego and Abd‑Almageed, Wael and Bestagini, Paolo and Tubaro, Stefano and Rocha, Anderson and Scheirer, Walter and Verdoliva, Luisa and Delp, Edward},
   title = {{SILA: a system for scientifc image analysis}},
   journal = {Nature Scientific Reports},
   year = 2022,
   number = {12},
   volume = {18306},
   pages = {1--15}
}
```
