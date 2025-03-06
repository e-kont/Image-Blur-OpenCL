__kernel void blur(__global const unsigned char* img_in, 
__global unsigned char* img_out, const int axis, __global const float* weight) 
{	
	const int KERNEL_RADIUS = 8;	
	// Get width and height value	
	int height = get_global_size(0);	
	int width = get_global_size(1);	
	int y = get_global_id(0);	
	int x = get_global_id(1);	
	int pixel = y * width + x;	
	for (int channel = 0; channel < 4; channel++) {	
		float sum_weight = 0.0f;
		float ret = 0.f;	
		for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++)	
		{	
			int offset_x = axis == 0 ? offset : 0;	
			int offset_y = axis == 1 ? offset : 0;	
			int pixel_y = max(min(y + offset_y, height - 1), 0);	
			int pixel_x = max(min(x + offset_x, width - 1), 0);	
			int pixel = pixel_y * width + pixel_x;
			ret += weight[offset + KERNEL_RADIUS] * img_in[4 * pixel + channel];
			sum_weight += weight[offset + KERNEL_RADIUS];		
		}	
		ret /= sum_weight;	
		img_out[4 * pixel + channel] = (unsigned char)max(min(ret, 255.f), 0.f);	
	}	
}