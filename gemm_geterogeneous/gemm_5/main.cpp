#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <chrono>
#include "opencl_gemm.h"
#include <cassert>

//[n * m] X [m * k] = [n * k]
std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());

template<typename T>
void generate_matrix(T* a, T* b, int n, int m, int k){
	std::uniform_real_distribution<> dis(-100, 100);
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
	
	if (ma > std::numeric_limits<T>::epsilon() * 1000) {
		std::cout << "threshold = \t\t" << ma << '\n';
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

int n = 32 * 200, m = 16 * 100, k = 16 * 100;

template<typename T>
void lets_go(const char * message){
	std::cout << message << '\n';
	char* filename;
	char* kernelname;
	char* kernelname_block;
	char* kernelname_image;
	if (message == "FLOAT") {
		filename = (char*)"gemm_float.cl";
		kernelname = (char*)"gemmFloat";
		kernelname_block = (char*)"gemmFloatBlock";
		kernelname_image = (char*)"gemmFloatImage";
	}
	else {
		filename = (char*)"gemm_double.cl";
		kernelname = (char*)"gemmDouble";
		kernelname_block = (char*)"gemmDoubleBlock";
	}
	
	T* a = new T[n * m];
	T* b = new T[m * k];
	T* c_seq = new T[n * k];
	T* c_opencl_cpu = new T[n * k];
	T* c_opencl_gpu = new T[n * k];
	T* c_opencl_cpu_gpu = new T[n * k];

	double cpu_time, gpu_time, cpu_gpu_time;
	/*auto seq_time = stupid_gemm(n, m, k, a, b, c_seq);
	std::cout << "sequential gemm = \t\t" << seq_time << '\n';*/

	std::cout << "Intel(R) Core(TM) i5-7500 VS NVIDIA GeForce GTX 1050 Ti: standart version\n******************************************************************\n";
	generate_matrix(a, b, n, m, k);
	cpu_time = opencl_gemm(2, 1, n, m, k, a, b, c_opencl_cpu, filename, kernelname, 1);
	// check_gemm(n, k, c_seq, c_opencl_cpu);
	std::cout << "cpu time = \t\t\t" << cpu_time << '\n';
	generate_matrix(a, b, n, m, k);
	gpu_time = opencl_gemm(2, 1, n, m, k, a, b, c_opencl_gpu, filename, kernelname, 0);
	// check_gemm(n, k, c_seq, c_opencl_gpu);
	std::cout << "gpu time = \t\t\t" << gpu_time << '\n';
	generate_matrix(a, b, n, m, k);
	cpu_gpu_time = opencl_gemm(2, 1, n, m, k, a, b, c_opencl_cpu_gpu, filename, kernelname, 0.0025);
	// check_gemm(n, k, c_seq, c_opencl_cpu_gpu);
	std::cout << "0.0025 * cpu & 0.9975 * gpu = \t" << cpu_gpu_time << '\n';


	std::cout << "\nIntel(R) Core(TM) i5-7500 VS Intel(R) HD Graphics 630: standart version\n******************************************************************\n";
	generate_matrix(a, b, n, m, k);
	cpu_time = opencl_gemm(2, 0, n, m, k, a, b, c_opencl_cpu, filename, kernelname, 1);
	// check_gemm(n, k, c_seq, c_opencl_cpu);
	std::cout << "cpu time = \t\t\t" << cpu_time << '\n';
	generate_matrix(a, b, n, m, k);
	gpu_time = opencl_gemm(2, 0, n, m, k, a, b, c_opencl_gpu, filename, kernelname, 0);
	// check_gemm(n, k, c_seq, c_opencl_gpu);
	std::cout << "gpu time = \t\t\t" << gpu_time << '\n';
	generate_matrix(a, b, n, m, k);
	cpu_gpu_time = opencl_gemm(2, 0, n, m, k, a, b, c_opencl_cpu_gpu, filename, kernelname, 0.0025);
	// check_gemm(n, k, c_seq, c_opencl_cpu_gpu);
	std::cout << "0.0025 * cpu & 0.9975 * gpu = \t" << cpu_gpu_time << '\n';



	std::cout << "\nIntel(R) Core(TM) i5-7500 VS NVIDIA GeForce GTX 1050 Ti: block version\n******************************************************************\n";
	generate_matrix(a, b, n, m, k);
	cpu_time = opencl_gemm(2, 1, n, m, k, a, b, c_opencl_cpu, filename, kernelname_block, 1);
	// check_gemm(n, k, c_seq, c_opencl_cpu);
	std::cout << "cpu time = \t\t\t" << cpu_time << '\n';
	generate_matrix(a, b, n, m, k);
	gpu_time = opencl_gemm(2, 1, n, m, k, a, b, c_opencl_gpu, filename, kernelname_block, 0);
	// check_gemm(n, k, c_seq, c_opencl_gpu);
	std::cout << "gpu time = \t\t\t" << gpu_time << '\n';
	generate_matrix(a, b, n, m, k);
	cpu_gpu_time = opencl_gemm(2, 1, n, m, k, a, b, c_opencl_cpu_gpu, filename, kernelname_block, 0.0025);
	// check_gemm(n, k, c_seq, c_opencl_cpu_gpu);
	std::cout << "0.0025 * cpu & 0.9975 * gpu = \t" << cpu_gpu_time << '\n';

	std::cout << "\nIntel(R) Core(TM) i5-7500 VS Intel(R) HD Graphics 630: block version\n******************************************************************\n";
	generate_matrix(a, b, n, m, k);
	cpu_time = opencl_gemm(2, 0, n, m, k, a, b, c_opencl_cpu, filename, kernelname_block, 1);
	// check_gemm(n, k, c_seq, c_opencl_cpu);
	std::cout << "cpu time = \t\t\t" << cpu_time << '\n';
	generate_matrix(a, b, n, m, k);
	gpu_time = opencl_gemm(2, 0, n, m, k, a, b, c_opencl_gpu, filename, kernelname_block, 0);
	// check_gemm(n, k, c_seq, c_opencl_gpu);
	std::cout << "gpu time = \t\t\t" << gpu_time << '\n';
	generate_matrix(a, b, n, m, k);
	cpu_gpu_time = opencl_gemm(2, 0, n, m, k, a, b, c_opencl_cpu_gpu, filename, kernelname_block, 0.9975);
	// check_gemm(n, k, c_seq, c_opencl_cpu_gpu);
	std::cout << "0.9975 * cpu & 0.0025 * gpu = \t" << cpu_gpu_time << "\n\n";

}

int main(){
	lets_go<float>("FLOAT");
	lets_go<double>("DOUBLE");
}

