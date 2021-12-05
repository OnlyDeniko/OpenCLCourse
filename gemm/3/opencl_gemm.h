#include <CL/cl.h>
#include <istream>
#include <fstream>

#define BLOCK_SIZE 16

void initialize(int platform_index, cl_device_id& device){
	cl_platform_id* platforms = new cl_platform_id[3];
	clGetPlatformIDs(3, platforms, nullptr);

	cl_device_id* devices = new cl_device_id[1];
	clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 1, devices, nullptr);
	device = devices[0];
}

char* get_device_name(cl_device_id& device){
	char ans[128];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 128, ans, nullptr);
	return ans;
}

std::string read_kernel(char* filename){
	std::ifstream is(filename);
	std::string ans, tmp;
	while(getline(is, tmp)){
		ans += tmp;
		ans += '\n';
	}
	return ans;
}

void check_ret(cl_int ret, const char* message){
	if (ret != CL_SUCCESS) {
		std::cout << message << '\n';
		std::cout << "RETCODE = " << ret << '\n';
		exit(1);
	}
}

template<typename T>
double opencl_gemm(int platform_index, int n, int m, int k, T* a, T* b, T* c, char* filename, char* kernelname){
	// std::cout << filename << ' ' << kernelname << '\n';
	cl_device_id device;
	initialize(platform_index, device);
	// std::cout << get_device_name(device) << '\n';
	std::string kernel_code = read_kernel(filename);
	// std::cout << kernel_code << '\n';
	size_t kernel_len = kernel_code.size();

	cl_int ret;
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	check_ret(ret, "create context");
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, nullptr, &ret);
	check_ret(ret, "create command queue");
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, &kernel_len, &ret);
	check_ret(ret, "create program");
	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	/*size_t logSize = 1000, actualLogSize;
	char *log = new char[logSize];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, &actualLogSize);
	printf("\n-------------------------------------\n");
	printf("log:\n%s", log);
	printf("-------------------------------------\n\n");*/

	check_ret(ret, "build program");
	cl_kernel kernel = clCreateKernel(program, kernelname, &ret);
	check_ret(ret, "create kernel");

	cl_mem memObjA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * n * m, nullptr, &ret);
	check_ret(ret, "create buffer A");
	cl_mem memObjB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * m * k, nullptr, &ret);
	check_ret(ret, "create buffer B");
	cl_mem memObjC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * n * k, nullptr, &ret);
	check_ret(ret, "create buffer C");

	ret = clEnqueueWriteBuffer(command_queue, memObjA, CL_TRUE, 0, sizeof(T) * n * m, a, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer A");
	ret = clEnqueueWriteBuffer(command_queue, memObjB, CL_TRUE, 0, sizeof(T) * m * k, b, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer B");

	ret = clSetKernelArg(kernel, 0, sizeof(int), &n);
	check_ret(ret, "set kernel arg 0");
	ret = clSetKernelArg(kernel, 1, sizeof(int), &m);
	check_ret(ret, "set kernel arg 1");
	ret = clSetKernelArg(kernel, 2, sizeof(int), &k);
	check_ret(ret, "set kernel arg 2");
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjA);
	check_ret(ret, "set kernel arg 3");
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjB);
	check_ret(ret, "set kernel arg 4");
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjC);
	check_ret(ret, "set kernel arg 5");

	size_t global_work_size[2] = { n, k };
	size_t group_size[2] = { BLOCK_SIZE, BLOCK_SIZE };

	double time = omp_get_wtime();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, group_size, 0, nullptr, nullptr);
	clFinish(command_queue);
	time = omp_get_wtime() - time;
	check_ret(ret, "clEnqueueNDRangeKernel");

	ret = clEnqueueReadBuffer(command_queue, memObjC, CL_TRUE, 0, sizeof(T) * n * k, c, 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadBuffer");

	clReleaseMemObject(memObjA);
	clReleaseMemObject(memObjB);
	clReleaseMemObject(memObjC);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return time;
}


template<typename T>
double opencl_gemm_image(int platform_index, int n, int m, int k, T* a, T* b, T* c, char* filename, char* kernelname) {
	// std::cout << filename << ' ' << kernelname << '\n';
	cl_device_id device;
	initialize(platform_index, device);
	// std::cout << get_device_name(device) << '\n';
	std::string kernel_code = read_kernel(filename);
	// std::cout << kernel_code << '\n';
	size_t kernel_len = kernel_code.size();

	cl_int ret;
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	check_ret(ret, "create context");
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, nullptr, &ret);
	check_ret(ret, "create command queue");
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, &kernel_len, &ret);
	check_ret(ret, "create program");
	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	/*size_t logSize = 1000, actualLogSize;
	char *log = new char[logSize];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, &actualLogSize);
	printf("\n-------------------------------------\n");
	printf("log:\n%s", log);
	printf("-------------------------------------\n\n");*/

	check_ret(ret, "build program");
	cl_kernel kernel = clCreateKernel(program, kernelname, &ret);
	check_ret(ret, "create kernel");

	const cl_image_format format = { CL_R, CL_FLOAT };
	const cl_image_desc descA = { CL_MEM_OBJECT_IMAGE2D, m, n, 1, 1, 0, 0, 0, 0};
	const cl_image_desc descB = { CL_MEM_OBJECT_IMAGE2D, k, m, 1, 1, 0, 0, 0, 0};
	const cl_image_desc descC = { CL_MEM_OBJECT_IMAGE2D, k, n, 1, 1, 0, 0, 0, 0};

	cl_mem bufferA = clCreateImage(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, &format, &descA, (void *)a, &ret);
	check_ret(ret, "create buffer A");
	cl_mem bufferB = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &descB, (void*)b, &ret);
	check_ret(ret, "create buffer B");
	cl_mem bufferC = clCreateImage(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &format, &descC, (void*)c, &ret);
	check_ret(ret, "create buffer C");

	ret = clSetKernelArg(kernel, 0, sizeof(int), &n);
	check_ret(ret, "set kernel arg 0");
	ret = clSetKernelArg(kernel, 1, sizeof(int), &m);
	check_ret(ret, "set kernel arg 1");
	ret = clSetKernelArg(kernel, 2, sizeof(int), &k);
	check_ret(ret, "set kernel arg 2");
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferA);
	check_ret(ret, "set kernel arg 3");
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferB);
	check_ret(ret, "set kernel arg 4");
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufferC);
	check_ret(ret, "set kernel arg 5");

	size_t global_work_size[2] = { n, k };
	size_t group_size[2] = { BLOCK_SIZE, BLOCK_SIZE };

	double time = omp_get_wtime();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, group_size, 0, nullptr, nullptr);
	clFinish(command_queue);
	time = omp_get_wtime() - time;
	check_ret(ret, "clEnqueueNDRangeKernel");

	const size_t origin[] = {0, 0, 0};
	const size_t region[] = { k, n, 1 };

	ret = clEnqueueReadImage(command_queue, bufferC, CL_TRUE, origin, region, 0, 0, c, 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadImage");

	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return time;
}