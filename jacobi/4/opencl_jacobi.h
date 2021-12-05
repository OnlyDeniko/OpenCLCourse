#include <CL/cl.h>
#include <istream>
#include <fstream>

#define BLOCK_SIZE 64

void initialize(int platform_index, cl_device_id& device) {
	cl_platform_id* platforms = new cl_platform_id[3];
	clGetPlatformIDs(3, platforms, nullptr);

	cl_device_id* devices = new cl_device_id[1];
	clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 1, devices, nullptr);
	device = devices[0];
}

char* get_device_name(cl_device_id& device) {
	char ans[128];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 128, ans, nullptr);
	return ans;
}

std::string read_kernel(char* filename) {
	std::ifstream is(filename);
	std::string ans, tmp;
	while (getline(is, tmp)) {
		ans += tmp;
		ans += '\n';
	}
	return ans;
}

void check_ret(cl_int ret, const char* message) {
	if (ret != CL_SUCCESS) {
		std::cout << message << '\n';
		std::cout << "RETCODE = " << ret << '\n';
		exit(1);
	}
}

template<typename T>
double opencl_jacobi(int platform_index, int n, T* a, T* b, T* x0, T* x1, T* delta, char* filename, char* kernelname, T eps) {
	cl_device_id device;
	initialize(platform_index, device);
	std::string kernel_code = read_kernel(filename);
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

	cl_mem memObjA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * n * n, nullptr, &ret);
	check_ret(ret, "create buffer A");
	ret = clEnqueueWriteBuffer(command_queue, memObjA, CL_TRUE, 0, sizeof(T) * n * n, a, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer A");
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjA);
	check_ret(ret, "set kernel arg 0");

	cl_mem memObjB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * n, nullptr, &ret);
	check_ret(ret, "create buffer B");
	ret = clEnqueueWriteBuffer(command_queue, memObjB, CL_TRUE, 0, sizeof(T) * n, b, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer B");
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjB);
	check_ret(ret, "set kernel arg 1");

	cl_mem memObjX0 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * n, nullptr, &ret);
	check_ret(ret, "create buffer X0");
	ret = clEnqueueWriteBuffer(command_queue, memObjX0, CL_TRUE, 0, sizeof(T) * n, x0, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer X0");

	cl_mem memObjX1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * n, nullptr, &ret);
	check_ret(ret, "create buffer X1");
	ret = clEnqueueWriteBuffer(command_queue, memObjX1, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer X1");

	cl_mem memObjDelta = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * n, nullptr, &ret);
	check_ret(ret, "create buffer delta");
	ret = clEnqueueWriteBuffer(command_queue, memObjDelta, CL_TRUE, 0, sizeof(T) * n, delta, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer delta");
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjDelta);
	check_ret(ret, "set kernel arg 4");

	size_t global_work_size[1] =  { n };
	size_t group_size = BLOCK_SIZE;

	int nIter = 100;
	T numerator, denominator;
	double time = omp_get_wtime();
	cl_event event;
	do{
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjX0);
		check_ret(ret, "set kernel arg 2");
		ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjX1);
		check_ret(ret, "set kernel arg 3");

		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size, &group_size, 0, nullptr, &event);
		check_ret(ret, "clEnqueueNDRangeKernel");
		clWaitForEvents(1, &event);

		ret = clEnqueueReadBuffer(command_queue, memObjDelta, CL_TRUE, 0, sizeof(T) * n, delta, 0, nullptr, nullptr);
		check_ret(ret, "clEnqueueReadBuffer");

		numerator = 0;
		for(int i = 0;i < n;i++){
			numerator += fabs(delta[i]);
		}
		std::swap(x0, x1);
		std::swap(memObjX0, memObjX1);
	} while (nIter-- && numerator > eps);
	
	std::cout << "Iterations: " << 100 - nIter << '\n';
	std::cout << "Accuracy: " << numerator << '\n';

	ret = clEnqueueReadBuffer(command_queue, memObjX0, CL_TRUE, 0, sizeof(T) * n, x0, 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadBuffer");
	ret = clEnqueueReadBuffer(command_queue, memObjX1, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadBuffer");
	
	time = omp_get_wtime() - time;
	
	clFinish(command_queue);
	clReleaseMemObject(memObjA);
	clReleaseMemObject(memObjB);
	clReleaseMemObject(memObjX0);
	clReleaseMemObject(memObjX1);
	clReleaseMemObject(memObjDelta);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return time;
}

