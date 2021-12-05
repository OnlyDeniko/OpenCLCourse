#include <CL/cl.h>
#include <iostream>

int main() {
	cl_uint platformCount = 0;

	clGetPlatformIDs(0, NULL, &platformCount);
	cl_platform_id platform = NULL;

	std::cout << platformCount << std::endl;

	if (platformCount > 0) {
		cl_platform_id* platforms = new cl_platform_id[platformCount];
		clGetPlatformIDs(platformCount, platforms, NULL);

		for (cl_uint i = 0; i < platformCount; ++i) {
			platform = platforms[i];

			char platformName[128];
			clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platformName, nullptr);
			std::cout << platformName << std::endl;

			cl_uint deviceCount = 0;
			clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
			std::cout << "device count :    " << deviceCount << std::endl;

			cl_device_id device;
			if (deviceCount > 0) {
				cl_device_id* devices = new cl_device_id[deviceCount];

				clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, devices, &deviceCount);

				device = devices[0];

				for (cl_uint j = 0; j < deviceCount; ++j) {
					char deviceName[128];
					clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, NULL);
					std::cout << "device name :   " << deviceName << '\n';


					const char* source =
						"__kernel void test( __global int * message, const unsigned int count) { \n"\
						"printf(\"I am from %d block, %d thread (global index: %d)\\n\", (int)get_group_id(0), (int)get_local_id(0), (int)get_global_id(0));\n"
						"int id = get_global_id(0);		\n"\
						"if (id < count) message[id] += id;		\n"\
						"					\n"\
						"}";

					size_t source_size = strlen(source);

					cl_context context = clCreateContext(NULL, 1, &devices[j], NULL, NULL, NULL);
					cl_command_queue command_queue;
					command_queue = clCreateCommandQueueWithProperties(context, devices[j], 0, NULL);

					cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_size, NULL);
					std::cout << program << '\n';
					cl_int ret = clBuildProgram(program, 1, &devices[j], NULL, NULL, NULL);

					cl_kernel kernel = clCreateKernel(program, "test", NULL);
					std::cout << kernel << '\n';
					unsigned int memLenth = 25;
					cl_int* mem = new cl_int[memLenth];
					for (int w = 0; w < memLenth; ++w) {
						mem[w] = 0;
					}

					cl_mem memobj = NULL;
					memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, memLenth * sizeof(cl_int), NULL, &ret);

					clEnqueueWriteBuffer(command_queue, memobj, CL_TRUE, 0, memLenth * sizeof(cl_int), mem, 0, NULL, NULL);

					clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj);
					clSetKernelArg(kernel, 1, sizeof(unsigned int), &memLenth);

					size_t group;
					clGetKernelWorkGroupInfo(kernel, devices[j], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
					//std::cout << "GROUP  " << group << '\n';

					unsigned int count = 16;
					size_t global_work_size[1] = { 32 };
					size_t size = 2;

					clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, &size, 0, NULL, NULL);
					clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, memLenth * sizeof(float), mem, 0, NULL, NULL);

					for (int ii = 0; ii < memLenth; ++ii) {
						std::cout << mem[ii] << ' ';
					}
					std::cout << std::endl;

					clReleaseMemObject(memobj);
					clReleaseProgram(program);
					clReleaseKernel(kernel);
					clReleaseCommandQueue(command_queue);
					clReleaseContext(context);
				}
			}
		}
	}
	return 0;
}