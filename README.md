# interview_homework solution 
Python code to generate interview test data.

# Installation
This repo has been developed on top of python 3.11, so...

    conda create -n env1 python=3.11

you may wish to use your own conda env name, not "env1". Then, install stuff:

    conda activate env1
    pip install -r req.txt

Alternatively, one can install conda (version 23.3.1 is used in the original case), and then use the following incantation:

    conda env create -f environment.yml

# The homework problem image generation


10 Images was generated and saved in the directory:

    images

The task for the HW exercise was to find each spot in each of the test images. This is minimally the center of each spot, 
but extra credit would be the extent of the spot as well. 

# The homework Solution

Two different solution is provided. 

Solution 1 (wavelet_denoising.py): Wavelet transform is used for denoising and creating a grayscale mask, which is then thresholded 
to create a binary mask. SimpleBlobDetector_create is applied on the binary mask to find the coordinates of 
the centers.

Solution 2 (kmeans_clustering.py): This approach uses K-means clustering to create a mask and then uses Hough_Circles to find the centers. 

The answer to the homework problem is given by a zip file of a directory containing code, support files and annotated (the original image with the spot location overlay) image files.












