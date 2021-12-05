#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <chrono>
#include "openmp_gemm.h"
#include "opencl_gemm.h"
#include <cassert>

//[n * m] X [m * k] = [n * k]

template<typename T>
void generate_matrix(T* a, T* b, int n, int m, int k){
	std::mt19937 gen(123);
	// std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<> dis(0, 1);
	for(int i = 0;i < n;i++) for(int j = 0;j < m;j++){
		a[i * m + j] = dis(gen);
	}
	for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) {
		b[i * k + j] = dis(gen);
	}
}

template<typename T>
double stupid_gemm(int n, int m, int k, const T const * a, const T const * b, T* c) {
	double time = omp_get_wtime();
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			T res = 0;
			for (int kk = 0; kk < m; kk++) {
				res += a[i * m + kk] * b[kk * k + j];
			}
			c[i * k + j] = res;
		}
	}
	time = omp_get_wtime() - time;
	return time;
}

template<typename T>
bool check_gemm(int n, int k, const T const* _1, const T const * _2){
	T ma = 0;
	for(int i = 0;i < n * k;i++){
		ma += std::abs(_1[i] - _2[i]);
	}
	ma /= n * k;
	// std::cout << "threshold = \t\t" << ma << '\n';
	if (ma > std::numeric_limits<T>::epsilon() * 1000) {
		exit(1);
	}
	return 1;
}

template<typename T>
void print_matrix(int n, int m, T* matrix, const char* message){
	std::cout << message << '\n';
	for(int i = 0;i < n;i++){
		for(int j = 0;j < m;std::cout << ' ' << matrix[i * m + j], ++j){}
		std::cout << '\n';
	}
}

template<typename T>
void clear_matrix(int n, int m, T* matrix){
	for (int i = 0; i < n * m; i++) matrix[i] = T(0);
}

int n = 16 * 100, m = 16 * 100, k = 16 * 100;

template<typename T>
void lets_go(const char * message){
	std::cout << message << '\n';
	T* a = new T[n * m];
	T* b = new T[m * k];
	T* c_seq = new T[n * k];
	T* c_omp = new T[n * k];
	T* c_omp_block_1= new T[n * k];
	T* c_omp_block_2 = new T[n * k];
	T* c_opencl_gpu = new T[n * k];
	T* c_opencl_cpu = new T[n * k];
	T* c_opencl_gpu_block = new T[n * k];
	T* c_opencl_cpu_block = new T[n * k];
	T* c_opencl_gpu_image = new T[n * k];
	T* c_opencl_cpu_image = new T[n * k];
	T* c_opencl_hd = new T[n * k];
	T* c_opencl_hd_block = new T[n * k];
	T* c_opencl_hd_image = new T[n * k];
	clear_matrix(n, k, c_seq);
	clear_matrix(n, k, c_omp);
	clear_matrix(n, k, c_omp_block_1);
	clear_matrix(n, k, c_omp_block_2);
	clear_matrix(n, k, c_opencl_gpu);
	clear_matrix(n, k, c_opencl_cpu);
	clear_matrix(n, k, c_opencl_gpu_block);
	clear_matrix(n, k, c_opencl_cpu_block);
	clear_matrix(n, k, c_opencl_gpu_image);
	clear_matrix(n, k, c_opencl_cpu_image);
	clear_matrix(n, k, c_opencl_hd);
	clear_matrix(n, k, c_opencl_hd_block);
	clear_matrix(n, k, c_opencl_hd_image);

	generate_matrix(a, b, n, m, k);
	
	auto seq_time = stupid_gemm(n, m, k, a, b, c_seq);
	std::cout << "sequential gemm = \t\t" << seq_time << '\n';

	
	std::cout << "\nOPENMP\n******************************************************************\n";
	
	auto omp_time = omp_gemm(n, m, k, a, b, c_omp);
	std::cout << "omp gemm = \t\t\t" << omp_time << '\n';
	check_gemm(n, k, c_seq, c_omp);
	auto omp_block_1_time = omp_gemm_block_1(n, m, k, a, b, c_omp_block_1);
	std::cout << "omp gemm block (version 1) = \t" << omp_block_1_time << '\n';
	check_gemm(n, k, c_seq, c_omp_block_1);
	auto omp_block_2_time = omp_gemm_block_2(n, m, k, a, b, c_omp_block_2);
	std::cout << "omp gemm block (version 2) = \t" << omp_block_2_time << '\n';
	check_gemm(n, k, c_seq, c_omp_block_2);

	
	char* filename;
	char* filename_block;
	char* filename_image;
	char* kernelname;
	char* kernelname_block;
	char* kernelname_image;
	if (message == "FLOAT"){
		filename = (char*)"gemm_float.cl";
		kernelname = (char*)"gemmFloat";
		filename_block = (char*)"gemm_block_float.cl";
		kernelname_block = (char*)"gemmFloatBlock";
		filename_image = (char*)"gemm_image_float.cl";
		kernelname_image = (char*)"gemmFloatImage";
	} else {
		filename = (char*)"gemm_double.cl";
		kernelname = (char*)"gemmDouble";
		filename_block = (char*)"gemm_block_double.cl";
		kernelname_block = (char*)"gemmDoubleBlock";
		filename_image = (char*)"gemm_image_float.cl";
		kernelname_image = (char*)"gemmFloatImage";
	}

	//HD Graphics
	std::cout << "\nHD GRAPHICS\n******************************************************************\n";
	auto opencl_hd_time = opencl_gemm(0, n, m, k, a, b, c_opencl_hd, filename, kernelname);
	std::cout << "opencl gemm hd = \t\t" << opencl_hd_time << '\n';
	check_gemm(n, k, c_omp_block_1, c_opencl_hd);
	auto opencl_hd_block_time = opencl_gemm(0, n, m, k, a, b, c_opencl_hd_block, filename_block, kernelname_block);
	std::cout << "opencl gemm block hd = \t\t" << opencl_hd_block_time << '\n';
	check_gemm(n, k, c_omp_block_1, c_opencl_hd_block);
	if (sizeof(a[0]) == 4) {
		auto opencl_hd_image_time = opencl_gemm_image(0, n, m, k, a, b, c_opencl_hd_image, filename_image, kernelname_image);
		std::cout << "opencl gemm image hd = \t\t" << opencl_hd_image_time << '\n';
		check_gemm(n, k, c_omp_block_1, c_opencl_hd_image);
	}

	//CPU
	std::cout << "\nCPU\n******************************************************************\n";
	auto opencl_cpu_time = opencl_gemm(2, n, m, k, a, b, c_opencl_cpu, filename, kernelname);
	std::cout << "opencl gemm cpu = \t\t" << opencl_cpu_time << '\n';
	check_gemm(n, k, c_omp_block_1, c_opencl_cpu);
	auto opencl_cpu_block_time = opencl_gemm(2, n, m, k, a, b, c_opencl_cpu_block, filename_block, kernelname_block);
	std::cout << "opencl gemm block cpu = \t" << opencl_cpu_block_time << '\n';
	check_gemm(n, k, c_omp_block_1, c_opencl_cpu_block);
	if (sizeof(a[0]) == 4){
		auto opencl_cpu_image_time = opencl_gemm_image(2, n, m, k, a, b, c_opencl_cpu_image, filename_image, kernelname_image);
		std::cout << "opencl gemm image cpu = \t" << opencl_cpu_image_time << '\n';
		check_gemm(n, k, c_omp_block_1, c_opencl_cpu_image);
	}

	//GPU
	std::cout << "\nGPU\n******************************************************************\n";
	auto opencl_gpu_time = opencl_gemm(1, n, m, k, a, b, c_opencl_gpu, filename, kernelname);
	std::cout << "opencl gemm gpu = \t\t" << opencl_gpu_time << '\n';
	check_gemm(n, k, c_omp_block_1, c_opencl_gpu);
	auto opencl_gpu_block_time = opencl_gemm(1, n, m, k, a, b, c_opencl_gpu_block, filename_block, kernelname_block);
	std::cout << "opencl gemm block gpu = \t" << opencl_gpu_block_time << '\n';
	check_gemm(n, k, c_omp_block_1, c_opencl_gpu_block);
	if (sizeof(a[0]) == 4){
		auto opencl_gpu_image_time = opencl_gemm_image(1, n, m, k, a, b, c_opencl_gpu_image, filename_image, kernelname_image);
		std::cout << "opencl gemm image gpu = \t" << opencl_gpu_image_time << '\n';
		check_gemm(n, k, c_omp_block_1, c_opencl_gpu_image);
	}
	std::cout << "******************************************************************\n";
}

int main(){
	lets_go<float>("FLOAT");
	lets_go<double>("DOUBLE");
}

