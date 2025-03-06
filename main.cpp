#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "CL\cl.h"
#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int KERNEL_RADIUS = 8;
const float sigma = 3.f;

void gaussian_blur_separate_parallel(const char* filename) {
	// Load image
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;

	// Allocate space for input, output images and blur weights in the host
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels, 4);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return;
	}

	unsigned char* img_out = new unsigned char[width * height * 4];
	unsigned char* img_horizontal_blur = new unsigned char[width * height * 4];
	float* weight = new float[2 * KERNEL_RADIUS + 1]; // Weights in the (-KERNEL_RADIUS, KERNEL_RADIUS) range

	// Calculate weights
	for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++) {
		weight[offset + KERNEL_RADIUS] = std::exp(-(offset * offset) / (2.f * sigma * sigma));
	}

	// Get platform and device information
	cl_platform_id* platforms = NULL;
	cl_uint num_platforms;
	// Set up the Platform
	cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
	// Get the devices list and choose a GPU device
	cl_device_id* device_list = NULL;
	cl_uint num_devices;
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
	// Create one OpenCL context for each device in the platform
	cl_context context;
	context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_list[0], 0, &clStatus);

	// Create memory buffers on the device for each image and blur weights
	cl_mem in_clmem /*Input image*/ = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &clStatus);
	cl_mem h_clmem /*Horizontally blurred image*/ = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * 4 * sizeof(unsigned char), NULL, &clStatus);
	cl_mem out_clmem /*Final image*/ = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &clStatus);
	cl_mem w_clmem /*Blur weights*/ = clCreateBuffer(context, CL_MEM_READ_ONLY, (2 * KERNEL_RADIUS + 1) * sizeof(float), NULL, &clStatus);

	// Copy the Buffers for images and weights to the device
	clStatus = clEnqueueWriteBuffer(command_queue, in_clmem, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), (void*)img_in, 0, NULL, NULL);
	clStatus = clEnqueueWriteBuffer(command_queue, h_clmem, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), (void*)img_horizontal_blur, 0, NULL, NULL);
	clStatus = clEnqueueWriteBuffer(command_queue, out_clmem, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), (void*)img_out, 0, NULL, NULL);
	clStatus = clEnqueueWriteBuffer(command_queue, w_clmem, CL_TRUE, 0, (2 * KERNEL_RADIUS + 1) * sizeof(float), (void*)weight, 0, NULL, NULL);

	// Stringify kernel code from .cl file
	FILE* fp = fopen("kernel.cl", "rb");
	if (!fp) {
		printf("Could not load kernel\n");
		return;
	}

	fseek(fp, 0, SEEK_END);
	size_t program_size = ftell(fp);
	rewind(fp);
	char* blur_kernel = (char*)malloc(program_size + 1);
	blur_kernel[program_size] = '\0';
	fread(blur_kernel, sizeof(char), program_size, fp);
	fclose(fp);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&blur_kernel, NULL, &clStatus);
	// Build the program
	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "blur", &clStatus);

	const int axis[2] = { 0, 1 }; // Values for horizontal and vertical axis

	// Set kernel arguments for horizontal blur
	clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_clmem); // Input image = initial image
	clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&h_clmem); // Output image = horizontally blurred image
	clStatus = clSetKernelArg(kernel, 2, sizeof(int), (void*)&axis[0]); // Axis = horizontal
	clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&w_clmem); // Weights only need to be passed once

	const size_t global_size[2] = { height, width }; // Two-dimensional size for nested for-loop
	const size_t local_work_size[2] = { height / 256, width / 256 };

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	// Execute kernel
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_work_size, 0, NULL, NULL);
	clStatus = clFinish(command_queue);
	// Read horizontally blurred image from buffer
	clStatus = clEnqueueReadBuffer(command_queue, h_clmem, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), (void*)img_horizontal_blur, 0, NULL, NULL);

	// Set kernel arguments for vertical blur
	clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&h_clmem); // Input image = horizontally blurred image
	clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_clmem); // Output image = final image
	clStatus = clSetKernelArg(kernel, 2, sizeof(int), (void*)&axis[1]); // Axis = vertical

	// Execute kernel
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_work_size, 0, NULL, NULL);
	clStatus = clFinish(command_queue);
	// Read final blurred image from buffer
	clStatus = clEnqueueReadBuffer(command_queue, out_clmem, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), (void*)img_out, 0, NULL, NULL);

	clStatus = clFlush(command_queue);
	clStatus = clFinish(command_queue);

	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur Separate - Parallel: Time %dms\n", time);

	// Write final image to file
	stbi_write_jpg("image_blurred_final.jpg", width, height, 4, img_out, 90);

	// Finally release all OpenCL allocated objects and host buffers.
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(in_clmem);
	clReleaseMemObject(h_clmem);
	clReleaseMemObject(out_clmem);
	clReleaseMemObject(w_clmem);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	stbi_image_free(img_in);
	delete[] weight;
	delete[] img_horizontal_blur;
	delete[] img_out;
}

int main(void) {
	const char* filename = "street_night.jpg";
	gaussian_blur_separate_parallel(filename);

	return 0;
}