# Image-Blur-OpenCL
Parallel Gaussian image blurring using OpenCL C.

# Included files

## street_night.jpg
Initial image expected to be blurred with gaussian_blur_separate_parallel() function.

## stb_image.h, stb_image_write.h
C++ libraries for reading and writing images.

## main.cpp
Main code to be executed.

## kernel.cl
Kernel function executed by OpenCL context.

# Functions

## gaussian_blur_separate_parallel()
Loads input image into pixel array (width * height * channels). Creates OpenCL context from a GPU device, initiates buffers and loads them with input and output images. Sets width and height as two-dimensional global size for nested for-loop and splits into optimal local working size for pixel parallelization. Result is written into a new image file named "image_blurred_final.jpg".

## blur()
The function that applies the Gaussian blur filter, horizontally or vertically, on each pixel. Parameters are an input image pointer, an output image pointer, the blur axis, and the blur weight. A single pixel is identified by local height (y) and width (x) values and blurring is performed for each channel (Red, Green, Blue, Alpha).
