#include <CL/cl.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>

const char* saxpy_kernel = 
"__kernel void saxpy(int n, float a, __global float* x, int incx, __global float* y, int incy){										\n" \
"int index = get_global_id(0);																										\n" \
"if (index < n) {																													\n" \
"	y[index * incy] += a * x[index * incx];																							\n" \
"}}";

const char* daxpy_kernel =
"__kernel void daxpy(int n, double a, __global double* x, int incx, __global double* y, int incy){										\n" \
"int index = get_global_id(0);																											\n" \
"if (index < n) {																														\n" \
"	y[index * incy] += a * x[index * incx];																								\n" \
"}}";


double saxpy_cpu(int n, float a, float* x, int incx, float* y, int incy) {
	double start = omp_get_wtime();
	for (int i = 0; i < n; ++i) {
		y[i * incy] += a * x[i * incx];
	}
	start = omp_get_wtime() - start;
	return start;
}

double daxpy_cpu(int n, double a, double* x, int incx, double* y, int incy) {
	double start = omp_get_wtime();
	for (int i = 0; i < n; ++i) {
		y[i * incy] += a * x[i * incx];
	}
	start = omp_get_wtime() - start;
	return start;
}

double saxpy_omp(int n, float a, float* x, int incx, float* y, int incy) {
	double start = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp for 
		for (int i = 0; i < n; ++i) {
			y[i * incy] += a * x[i * incx];
		}
	}
	
	start = omp_get_wtime() - start;
	return start;
}


double daxpy_omp(int n, double a, double* x, int incx, double* y, int incy) {
	double start = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp for 
		for (int i = 0; i < n; ++i) {
			y[i * incy] += a * x[i * incx];
		}
	}

	start = omp_get_wtime() - start;
	return start;
}

template<typename T>
double run_opencl_kernel(cl_device_id& device, const char* source, const char* kernel_name, int len, T a, T* x, int incx, T* y, int incy, size_t group = 16){
	size_t source_size = strlen(source);
	cl_int ret;
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	/*cl_queue_properties prop[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, prop, &ret);*/
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, nullptr, &ret);

	cl_program program = clCreateProgramWithSource(context, 1, &source, &source_size, &ret);

	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
	
	cl_kernel kernel = clCreateKernel(program, kernel_name, &ret);
	
	cl_mem memObjX = clCreateBuffer(context, CL_MEM_READ_ONLY, len * sizeof(T), nullptr, &ret);
	cl_mem memObjY = clCreateBuffer(context, CL_MEM_READ_WRITE, len * sizeof(T), nullptr, &ret);

	ret = clEnqueueWriteBuffer(command_queue, memObjX, CL_TRUE, 0, len * sizeof(T), x, 0, nullptr, nullptr);

	ret = clSetKernelArg(kernel, 0, sizeof(int), &len);
	ret = clSetKernelArg(kernel, 1, sizeof(T), &a);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjX);
	ret = clSetKernelArg(kernel, 3, sizeof(int), &incx);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjY);
	ret = clSetKernelArg(kernel, 5, sizeof(int), &incy);

	// size_t work_size = (10000000 + 255) / 256 * 256;
	size_t work_size = (len + group - 1) / group * group;

	size_t global_work_size[1] = { work_size };

	double start = omp_get_wtime();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size, &group, 0, nullptr, nullptr);
	/*cl_event event;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size, &group, 0, nullptr, &event);
	clWaitForEvents(1, &event);*/
	clFinish(command_queue);
	start = omp_get_wtime() - start;
	ret = clEnqueueReadBuffer(command_queue, memObjY, CL_TRUE, 0, len * sizeof(T), y, 0, nullptr, nullptr);

	clReleaseMemObject(memObjX);
	clReleaseMemObject(memObjY);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	

	/*cl_ulong time_start;
	cl_ulong time_end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	clReleaseEvent(event);*/

	// return (double)(time_end - time_start) / 1e9 * work_size / 1e5;
	return start;
}


template<typename T>
void generator(std::vector<T> & a, int & n, T & A, bool generate_len){
	// std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	std::mt19937 gen(123);
	std::uniform_real_distribution<> dis(-1e2, 1e2);
	A = dis(gen);
	if (generate_len) n = gen() % 1000000 + 1;
	a.resize(n);
	for(int i = 0;i < n;i++){
		a[i] = dis(gen);
	}
}


template<class T>
void check(std::vector<T> & a, std::vector<T> & b){
	/*for (auto i : a) std::cout << i << ' ';
	std::cout << '\n';
	for (auto i : b) std::cout << i << ' ';
	std::cout << '\n';*/
	for(int i = 0;i < a.size();++i){
		if (std::fabs(a[i] - b[i]) > std::numeric_limits<T>::epsilon()) {
			std::cout << std::fabs(a[i] - b[i]) << '\n';
			exit(1);
		}
	}
}


void test_precision(){
	cl_uint platformCount = 0;

	clGetPlatformIDs(0, nullptr, &platformCount);

	cl_platform_id* platforms = new cl_platform_id[platformCount];
	clGetPlatformIDs(platformCount, platforms, nullptr);
	const int tests = 10;
	for (cl_uint i = 0; i < platformCount; ++i) {
		char platformName[128];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
		// std::cout << platformName << '\n';

		cl_uint deviceCount = 0;
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

		cl_device_id* devices = new cl_device_id[deviceCount];

		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1, devices, &deviceCount);

		for (cl_uint j = 0; j < deviceCount; ++j) {
			char deviceName[128];
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, nullptr);
			// std::cout << "device name: " << deviceName << '\n';
			
			for (int test = 0; test < tests; ++test) {
				std::vector<float> x;
				int len;
				float a;
				generator(x, len, a, true);
				std::vector<float> true_y(len, 0), y(len, 0);

				saxpy_cpu(len, a, x.data(), 1, true_y.data(), 1);
				double opencl_time = run_opencl_kernel(devices[j], saxpy_kernel, "saxpy", len, a, x.data(), 1, y.data(), 1);
				check(true_y, y);
				std::cout << "test opencl float #" << test << " done\n";
			}
			for (int test = 0; test < tests; ++test) {
				std::vector<double> x;
				int len;
				double a;
				generator(x, len, a, true);
				std::vector<double> true_y(len, 0), y(len, 0);

				daxpy_cpu(len, a, x.data(), 1, true_y.data(), 1);
				double opencl_time = run_opencl_kernel(devices[j], daxpy_kernel, "daxpy", len, a, x.data(), 1, y.data(), 1);
				check(true_y, y);
				std::cout << "test opencl double #" << test << " done\n";
			}
		}
	}

	for(int test = 0;test < tests;test++){
		std::vector<float> x;
		int len;
		float a;
		generator(x, len, a, true);
		std::vector<float> true_y(len, 0), y(len, 0);

		saxpy_cpu(len, a, x.data(), 1, true_y.data(), 1);
		saxpy_omp(len, a, x.data(), 1, y.data(), 1);
		check(true_y, y);
		std::cout << "test openmp float #" << test << " done\n";
	}
	for (int test = 0; test < tests; test++) {
		std::vector<double> x;
		int len;
		double a;
		generator(x, len, a, true);
		std::vector<double> true_y(len, 0), y(len, 0);

		daxpy_cpu(len, a, x.data(), 1, true_y.data(), 1);
		daxpy_omp(len, a, x.data(), 1, y.data(), 1);
		check(true_y, y);
		std::cout << "test openmp double #" << test << " done\n";
	}

}


void performance(){
	cl_uint platformCount = 0;

	clGetPlatformIDs(0, nullptr, &platformCount);

	cl_platform_id* platforms = new cl_platform_id[platformCount];
	clGetPlatformIDs(platformCount, platforms, nullptr);
	const int start_len = 1e5;
	const int finish_len = 1e5;
	const int step_len = 1e5;
	for (cl_uint i = 0; i < 3; ++i) {
		char platformName[128];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
		// std::cout << platformName << '\n';

		cl_uint deviceCount = 0;
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

		cl_device_id* devices = new cl_device_id[deviceCount];

		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1, devices, &deviceCount);
		
		for (cl_uint j = 0; j < deviceCount; ++j) {
			char deviceName[128];
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, nullptr);
			std::cout << std::string("OpenCL ") + deviceName << '\n';
			
			for (size_t group = 8; group <= 256; group <<= 1) {
				for (int len = start_len; len < finish_len; len += step_len) {
					std::vector<float> x;
					float a;
					generator(x, len, a, false);
					std::vector<float> y(len, 0);

					double opencl_time = run_opencl_kernel(devices[j], saxpy_kernel, "saxpy", len, a, x.data(), 1, y.data(), 1);
					std::cout << "float n " << len << " group " << group << " time " << std::fixed << std::setprecision(20) << opencl_time << '\n';
				}
			}
			for (size_t group = 8; group <= 256; group <<= 1) {
				for (int len = start_len; len < finish_len; len += step_len) {
					std::vector<double> x;
					double a;
					generator(x, len, a, false);
					std::vector<double> y(len, 0);

					double opencl_time = run_opencl_kernel(devices[j], daxpy_kernel, "daxpy", len, a, x.data(), 1, y.data(), 1);
					std::cout << "double n " << len << " group " << group << " time " << std::fixed << std::setprecision(20) << opencl_time << '\n';
				}
			}
		}
	}
	std::cout << "OpenMP\n";
	for (size_t group = 8; group <= 256; group <<= 1) {
		for (int len = start_len; len < finish_len; len += step_len) {
			std::vector<float> x;
			float a;
			generator(x, len, a, false);
			std::vector<float> y(len, 0);

			double openmp_time = saxpy_omp(len, a, x.data(), 1, y.data(), 1);
			std::cout << "float n " << len << " group " << group << " time " << std::fixed << std::setprecision(20) << openmp_time << '\n';
		}
	}
	for (size_t group = 8; group <= 256; group <<= 1) {
		for (int len = start_len; len < finish_len; len += step_len) {
			std::vector<double> x;
			double a;
			generator(x, len, a, false);
			std::vector<double> y(len, 0);

			double openmp_time = daxpy_omp(len, a, x.data(), 1, y.data(), 1);
			std::cout << "double n " << len << " group " << group << " time " << std::fixed << std::setprecision(20) << openmp_time << '\n';
		}
	}
	std::cout << "Sequential CPU\n";
	for (size_t group = 8; group <= 256; group <<= 1) {
		for (int len = start_len; len < finish_len; len += step_len) {
			std::vector<float> x;
			float a;
			generator(x, len, a, false);
			std::vector<float> y(len, 0);

			double cpu_time = saxpy_cpu(len, a, x.data(), 1, y.data(), 1);
			std::cout << "float n " << len << " group " << group << " time " << std::fixed << std::setprecision(20) << cpu_time << '\n';
		}
	}
	for (size_t group = 8; group <= 256; group <<= 1) {
		for (int len = start_len; len < finish_len; len += step_len) {
			std::vector<double> x;
			double a;
			generator(x, len, a, false);
			std::vector<double> y(len, 0);

			double cpu_time = daxpy_cpu(len, a, x.data(), 1, y.data(), 1);
			std::cout << "double n " << len << " group " << group << " time " << std::fixed << std::setprecision(20) << cpu_time << '\n';
		}
	}
}


int main() {
	// freopen("output.txt", "w", stdout);
	
	// test_precision();

	performance();

	return 0;
}