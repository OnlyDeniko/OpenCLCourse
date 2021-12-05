#include <CL/cl.h>
#include <istream>
#include <fstream>

#define BLOCK_SIZE 16

void get_device_name(cl_device_id& device) {
	char ans[128];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 128, ans, nullptr);
	// std::cout << ans << '\n';
}

void initialize(int platform_index, cl_device_id& device){
	cl_platform_id* platforms = new cl_platform_id[3];
	clGetPlatformIDs(3, platforms, nullptr);

	cl_device_id* devices = new cl_device_id[1];
	clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 1, devices, nullptr);
	device = devices[0];
	get_device_name(device);
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

struct opencl_env {
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem memObjA, memObjB, memObjC;
	opencl_env(
		cl_context _context,
		cl_command_queue _queue,
		cl_program _program,
		cl_kernel _kernel,
		cl_mem _memA,
		cl_mem _memB,
		cl_mem _memC
	) : context(_context), queue(_queue), program(_program), kernel(_kernel), memObjA(_memA), memObjB(_memB), memObjC(_memC) {}
	~opencl_env(){
		clReleaseMemObject(memObjA);
		clReleaseMemObject(memObjB);
		clReleaseMemObject(memObjC);
		clReleaseProgram(program);
		clReleaseKernel(kernel);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
	}
};

template<typename T>
opencl_env create_env(
		int platform_index,
		int n,
		int m,
		int k,
		T* a,
		T* b,
		char* filename,
		char* kernelname){
	cl_device_id device;
	initialize(platform_index, device);
	std::string kernel_code = read_kernel(filename);
	size_t kernel_len = kernel_code.size();

	cl_int ret;
	// Create context
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	check_ret(ret, "clCreateContext");
	cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	// Create queue
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, props, &ret);
	check_ret(ret, "clCreateCommandQueueWithProperties");
	// Create program
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, &kernel_len, &ret);
	check_ret(ret, "clCreateProgramWithSource");
	// Build
	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
	check_ret(ret, "clBuildProgram");
	// Create kernel
	cl_kernel kernel = clCreateKernel(program, kernelname, &ret);
	check_ret(ret, "clCreateKernel");
	// Create buffer
	cl_mem memObjA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * n * m, nullptr, &ret);
	check_ret(ret, "create buffer A");
	cl_mem memObjB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * m * k, nullptr, &ret);
	check_ret(ret, "create buffer B");
	cl_mem memObjC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * n * k, nullptr, &ret);
	check_ret(ret, "create buffer C");
	// Write buffer
	ret = clEnqueueWriteBuffer(command_queue, memObjA, CL_TRUE, 0, sizeof(T) * n * m, a, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer A");
	ret = clEnqueueWriteBuffer(command_queue, memObjB, CL_TRUE, 0, sizeof(T) * m * k, b, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer B");
	// Set kernel Args
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
	return opencl_env(context, command_queue, program, kernel, memObjA, memObjB, memObjC);
}

template<typename T>
double opencl_gemm(
		int cpu_platform_index,
		int gpu_platform_index,
		int n,
		int m,
		int k,
		T* a,
		T* b,
		T* c,
		char* filename,
		char* kernelname,
		double partition) {
	int cpu_n = n * partition;
	int gpu_n = n - cpu_n;

	if (cpu_n % BLOCK_SIZE != 0 || gpu_n % BLOCK_SIZE != 0){
		std::cout << "wrong size of matrices\n";
		exit(1);
	}
	size_t cpu_global_work_size[2] = { k, cpu_n };
	size_t gpu_global_work_size[2] = { k, gpu_n };
	size_t group_size[2] = { BLOCK_SIZE, BLOCK_SIZE };
	cl_int ret;

	if (gpu_n == 0){
		opencl_env cpu_env = create_env(cpu_platform_index, cpu_n, m, k, a, b, filename, kernelname);
		cl_event cpu_event;
		ret = clEnqueueNDRangeKernel(cpu_env.queue, cpu_env.kernel, 2, nullptr, cpu_global_work_size, group_size, 0, nullptr, &cpu_event);
		clWaitForEvents(1, &cpu_event);
		clFinish(cpu_env.queue);
		ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjC, CL_TRUE, 0, sizeof(T) * cpu_n * k, c, 0, nullptr, nullptr);
		cl_ulong time_start;
		cl_ulong time_end;

		clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

		double nanoSeconds = time_end - time_start;
		return nanoSeconds / 1e9;
	}

	if (cpu_n == 0){
		opencl_env gpu_env = create_env(gpu_platform_index, gpu_n, m, k, a, b, filename, kernelname);
		cl_event gpu_event;
		ret = clEnqueueNDRangeKernel(gpu_env.queue, gpu_env.kernel, 2, nullptr, gpu_global_work_size, group_size, 0, nullptr, &gpu_event);
		clWaitForEvents(1, &gpu_event);
		clFinish(gpu_env.queue);

		ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjC, CL_TRUE, 0, sizeof(T) * gpu_n * k, c, 0, nullptr, nullptr);
		cl_ulong time_start;
		cl_ulong time_end;

		clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

		double nanoSeconds = time_end - time_start;
		return nanoSeconds / 1e9;
	}

	opencl_env cpu_env = create_env(cpu_platform_index, cpu_n, m, k, a, b, filename, kernelname);
	opencl_env gpu_env = create_env(gpu_platform_index, gpu_n, m, k, &a[cpu_n * m], b, filename, kernelname);


	cl_event cpu_event, gpu_event;
	ret = clEnqueueNDRangeKernel(cpu_env.queue, cpu_env.kernel, 2, nullptr, cpu_global_work_size, group_size, 0, nullptr, &cpu_event);
	ret = clEnqueueNDRangeKernel(gpu_env.queue, gpu_env.kernel, 2, nullptr, gpu_global_work_size, group_size, 0, nullptr, &gpu_event);
	clWaitForEvents(1, &cpu_event);
	clWaitForEvents(1, &gpu_event);
	clFinish(cpu_env.queue);
	clFinish(gpu_env.queue);

	ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjC, CL_TRUE, 0, sizeof(T) * cpu_n * k, c, 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadBuffer");
	ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjC, CL_TRUE, 0, sizeof(T) * gpu_n * k, &c[cpu_n * k], 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadBuffer");

	cl_ulong time_start;
	cl_ulong time_end;

	clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

	double cpu_nanoSeconds = time_end - time_start;

	clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

	double gpu_nanoSeconds = time_end - time_start;

	return cpu_nanoSeconds / 1e9 + gpu_nanoSeconds / 1e9;
}
