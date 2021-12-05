#include <CL/cl.h>
#include <istream>
#include <fstream>
#include <omp.h>

#define BLOCK_SIZE 32

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


struct opencl_env {
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem memObjA, memObjB, memObjX0, memObjX1, memObjDelta;
	opencl_env(
		cl_context _context,
		cl_command_queue _queue,
		cl_program _program,
		cl_kernel _kernel,
		cl_mem _memA,
		cl_mem _memB,
		cl_mem _memX0,
		cl_mem _memX1,
		cl_mem _memDelta
	) : context(_context),
		queue(_queue),
		program(_program),
		kernel(_kernel),
		memObjA(_memA),
		memObjB(_memB),
		memObjX0(_memX0),
		memObjX1(_memX1),
		memObjDelta(_memDelta) {}
	~opencl_env() {
		clReleaseMemObject(memObjA);
		clReleaseMemObject(memObjB);
		clReleaseMemObject(memObjX0);
		clReleaseMemObject(memObjX1);
		clReleaseMemObject(memObjDelta);
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
	int stride,
	T* a,
	T* b,
	T* x0,
	T* x1,
	T* delta,
	char* filename,
	char* kernelname) {
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
	ret = clEnqueueWriteBuffer(command_queue, memObjA, CL_TRUE, 0, sizeof(T) * n * m, a, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer A");
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjA);
	check_ret(ret, "set kernel arg 0");

	cl_mem memObjB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * m, nullptr, &ret);
	check_ret(ret, "create buffer B");
	ret = clEnqueueWriteBuffer(command_queue, memObjB, CL_TRUE, 0, sizeof(T) * m, b, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer B");
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjB);
	check_ret(ret, "set kernel arg 1");

	cl_mem memObjX0 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * m, nullptr, &ret);
	check_ret(ret, "create buffer X0");
	ret = clEnqueueWriteBuffer(command_queue, memObjX0, CL_TRUE, 0, sizeof(T) * m, x0, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer X0");

	cl_mem memObjX1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * m, nullptr, &ret);
	check_ret(ret, "create buffer X1");
	ret = clEnqueueWriteBuffer(command_queue, memObjX1, CL_TRUE, 0, sizeof(T) * m, x1, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer X1");
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjX1);
	check_ret(ret, "set kernel arg 3");

	cl_mem memObjDelta = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * m, nullptr, &ret);
	check_ret(ret, "create buffer delta");
	ret = clEnqueueWriteBuffer(command_queue, memObjDelta, CL_TRUE, 0, sizeof(T) * m, delta, 0, nullptr, nullptr);
	check_ret(ret, "EnqueueWriteBuffer delta");
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjDelta);
	check_ret(ret, "set kernel arg 4");

	ret = clSetKernelArg(kernel, 5, sizeof(int), &m);
	check_ret(ret, "set kernel arg 5");

	ret = clSetKernelArg(kernel, 6, sizeof(int), &stride);
	check_ret(ret, "set kernel arg 6");
	return opencl_env(context, command_queue, program, kernel, memObjA, memObjB, memObjX0, memObjX1, memObjDelta);
}

template<typename T>
std::pair<double, double> opencl_jacobi(
		int cpu_platform_index,
		int gpu_platform_index,
		int n,
		T* a,
		T* b,
		T* x0,
		T* x1,
		T* delta,
		char* filename,
		char* kernelname,
		double partition,
		double eps,
		int nIter) {
	
	int cpu_n = n * partition;
	int gpu_n = n - cpu_n;
	
	if (cpu_n % BLOCK_SIZE != 0 || gpu_n % BLOCK_SIZE != 0){
		std::cout << "wrong size of matrices\n";
		exit(1);
	}

	size_t cpu_global_work_size[1] = { cpu_n };
	size_t gpu_global_work_size[1] = { gpu_n };
	size_t group_size = BLOCK_SIZE;
	cl_int ret;
	int iter = 0;
	T acc;
	double time = 0;
	cl_ulong time_start;
	cl_ulong time_end;
	cl_event cpu_event, gpu_event;
	T* buf = new T[n];
	double full_time = omp_get_wtime();

	if (gpu_n == 0){
		opencl_env cpu_env = create_env(cpu_platform_index, cpu_n, n, 0, a, b, x0, x1, delta, filename, kernelname);

		do {
			ret = clSetKernelArg(cpu_env.kernel, 2, sizeof(cl_mem), &cpu_env.memObjX0);
			check_ret(ret, "set cpu kernel arg 2");
			
			ret = clEnqueueNDRangeKernel(cpu_env.queue, cpu_env.kernel, 1, nullptr, cpu_global_work_size, &group_size, 0, nullptr, &cpu_event);
			clWaitForEvents(1, &cpu_event);

			clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
			clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
			time += (time_end - time_start) / 1e9;

			ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjDelta, CL_TRUE, 0, sizeof(T) * n, delta, 0, nullptr, nullptr);
			check_ret(ret, "clEnqueueReadBuffer");

			acc = 0;
			for (int i = 0; i < cpu_n; i++) {
				acc += fabs(delta[i]);
			}

			// x1 to x0
			ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjX1, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
			check_ret(ret, "clEnqueueReadBuffer");
			memcpy(x1, buf, sizeof(T) * cpu_n);

			ret = clEnqueueWriteBuffer(cpu_env.queue, cpu_env.memObjX0, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
			check_ret(ret, "EnqueueWriteBuffer X0");
		} while (iter++ < nIter && acc > eps);

		/*std::cout << "Iterations: " << iter << '\n';
		std::cout << "Accuracy: " << acc << '\n';*/

		ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjX0, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
		check_ret(ret, "clEnqueueReadBuffer");
		memcpy(x0, buf, sizeof(T) * cpu_n);

		ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjX1, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
		check_ret(ret, "clEnqueueReadBuffer");
		memcpy(x1, buf, sizeof(T) * cpu_n);
		
		clFinish(cpu_env.queue);
		full_time = omp_get_wtime() - full_time;
		return std::make_pair(time, full_time);
	}

	if (cpu_n == 0){
		opencl_env gpu_env = create_env(gpu_platform_index, gpu_n, n, cpu_n, &a[cpu_n * n], b, x0, x1, delta, filename, kernelname);

		do {
			ret = clSetKernelArg(gpu_env.kernel, 2, sizeof(cl_mem), &gpu_env.memObjX0);
			check_ret(ret, "set gpu kernel arg 2");

			ret = clEnqueueNDRangeKernel(gpu_env.queue, gpu_env.kernel, 1, nullptr, gpu_global_work_size, &group_size, 0, nullptr, &gpu_event);
			clWaitForEvents(1, &gpu_event);

			clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
			clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
			time += (time_end - time_start) / 1e9;

			ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjDelta, CL_TRUE, 0, sizeof(T) * n, delta, 0, nullptr, nullptr);
			check_ret(ret, "clEnqueueReadBuffer");

			acc = 0;
			for (int i = cpu_n; i < n; i++) {
				acc += fabs(delta[i]);
			}

			// x1 to x0
			ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjX1, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
			check_ret(ret, "clEnqueueReadBuffer");
			memcpy(x1 + cpu_n, buf + cpu_n, sizeof(T) * gpu_n);

			ret = clEnqueueWriteBuffer(gpu_env.queue, gpu_env.memObjX0, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
			check_ret(ret, "EnqueueWriteBuffer X0");
		} while (iter++ < nIter && acc > eps);

		/*std::cout << "Iterations: " << iter << '\n';
		std::cout << "Accuracy: " << acc << '\n';*/

		ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjX0, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
		check_ret(ret, "clEnqueueReadBuffer");
		memcpy(x0 + cpu_n, buf + cpu_n, sizeof(T) * gpu_n);

		ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjX1, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
		check_ret(ret, "clEnqueueReadBuffer");
		memcpy(x1 + cpu_n, buf + cpu_n, sizeof(T) * gpu_n);

		clFinish(gpu_env.queue);
		full_time = omp_get_wtime() - full_time;
		return std::make_pair(time, full_time);
	}

	opencl_env cpu_env = create_env(cpu_platform_index, cpu_n, n, 0, a, b, x0, x1, delta, filename, kernelname);
	opencl_env gpu_env = create_env(gpu_platform_index, gpu_n, n, cpu_n, &a[cpu_n * n], b, x0, x1, delta, filename, kernelname);

	
	do{
		ret = clSetKernelArg(cpu_env.kernel, 2, sizeof(cl_mem), &cpu_env.memObjX0);
		check_ret(ret, "set cpu kernel arg 2");
		ret = clSetKernelArg(gpu_env.kernel, 2, sizeof(cl_mem), &gpu_env.memObjX0);
		check_ret(ret, "set gpu kernel arg 2");

		ret = clEnqueueNDRangeKernel(cpu_env.queue, cpu_env.kernel, 1, nullptr, cpu_global_work_size, &group_size, 0, nullptr, &cpu_event);
		ret = clEnqueueNDRangeKernel(gpu_env.queue, gpu_env.kernel, 1, nullptr, gpu_global_work_size, &group_size, 0, nullptr, &gpu_event);
		clWaitForEvents(1, &cpu_event);
		clWaitForEvents(1, &gpu_event);

		clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
		clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
		time += (time_end - time_start) / 1e9;

		clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
		clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
		time += (time_end - time_start) / 1e9;

		ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjDelta, CL_TRUE, 0, sizeof(T) * n, delta, 0, nullptr, nullptr);
		check_ret(ret, "clEnqueueReadBuffer");

		acc = 0;
		for(int i = 0;i < cpu_n;i++){
			acc += fabs(delta[i]);
		}

		ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjDelta, CL_TRUE, 0, sizeof(T) * n, delta, 0, nullptr, nullptr);
		check_ret(ret, "clEnqueueReadBuffer");

		for (int i = cpu_n; i < n; i++) {
			acc += fabs(delta[i]);
		}
		
		// x1 to x0
		ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjX1, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
		check_ret(ret, "clEnqueueReadBuffer");
		memcpy(x1, buf, sizeof(T) * cpu_n);
		ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjX1, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
		check_ret(ret, "clEnqueueReadBuffer");
		memcpy(x1 + cpu_n, buf + cpu_n, sizeof(T) * gpu_n);
		
		ret = clEnqueueWriteBuffer(cpu_env.queue, cpu_env.memObjX0, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
		check_ret(ret, "EnqueueWriteBuffer X0");
		ret = clEnqueueWriteBuffer(gpu_env.queue, gpu_env.memObjX0, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
		check_ret(ret, "EnqueueWriteBuffer X0");
	} while (iter++ < nIter && acc > eps);
	
	/*std::cout << "Iterations: " << iter << '\n';
	std::cout << "Accuracy: " << acc << '\n';*/

	ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjX0, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadBuffer");
	memcpy(x0, buf, sizeof(T) * cpu_n);
	ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjX0, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadBuffer");
	memcpy(x0 + cpu_n, buf + cpu_n, sizeof(T) * gpu_n);

	ret = clEnqueueReadBuffer(cpu_env.queue, cpu_env.memObjX1, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadBuffer");
	memcpy(x1, buf, sizeof(T) * cpu_n);
	ret = clEnqueueReadBuffer(gpu_env.queue, gpu_env.memObjX1, CL_TRUE, 0, sizeof(T) * n, buf, 0, nullptr, nullptr);
	check_ret(ret, "clEnqueueReadBuffer");
	memcpy(x1 + cpu_n, buf + cpu_n, sizeof(T) * gpu_n);
	
	clFinish(cpu_env.queue);
	clFinish(gpu_env.queue);

	full_time = omp_get_wtime() - full_time;
	return std::make_pair(time, full_time);
}

