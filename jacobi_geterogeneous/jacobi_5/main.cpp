#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <chrono>
#include <cassert>
#include "opencl_jacobi.h"
std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
const int n = 32 * 500;

template<typename T>
void generateA(T * a){
	for(int i = 0;i < n;i++){
		for(int j = 0;j < n;j++){
			T tmp = gen() % 50000;
			tmp /= n;
			a[i * n + j] = (i == j ? tmp + 100000 : tmp);
		}
	}
}

template<typename T>
void generateB(T* b) {
	for (int i = 0; i < n; i++) {
		T tmp = gen();
		tmp /= n;
		b[i] = tmp;
	}
}

template<typename T>
bool check(T* a){
	for(int i = 0;i < n;i++){
		T sum = 0;
		for (int j = 0; j < n; j++) sum += a[i * n + j];
		sum -= a[i * n + i];
		if (sum > a[i * n + i]) return false;
	}
	return true;
}

template<typename T>
bool check_solution(T* a, T* b, T* x) {
	T* res = new T[n];
	for(int i = 0;i < n;i++){
		res[i] = 0;
		for(int j = 0;j < n;j++){
			res[i] += a[i * n + j] * x[j];
		}
	}
	T num = 0;
	for(int i = 0;i < n;i++){
		num += std::abs((res[i] - b[i]) / res[i]);
	}
	return num < (sizeof(a[0]) == 4 ? T(1) : T(1e-10));
}

template<typename T>
void print_matrix(int n, int m, T* matrix, const char* message) {
	std::cout << message << '\n';
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; std::cout << ' ' << matrix[i * m + j], ++j) {}
		std::cout << '\n';
	}
}

template<typename T>
void lets_go(const char* message) {
	std::cout << message << '\n';
	char* filename;
	char* kernelname;

	if (message == "FLOAT") {
		filename = (char*)"jacobi_float.cl";
		kernelname = (char*)"jacobiFloat";
	}
	else {
		filename = (char*)"jacobi_double.cl";
		kernelname = (char*)"jacobiDouble";
	}

	T* a = new T[n * n];
	T* b = new T[n];
	T* x0 = new T[n];
	T* x1 = new T[n];
	T* delta = new T[n];
	std::pair<double, double> cpu_time, gpu_time, cpu_gpu_time;
	generateA(a);

	if (!check(a)){
		std::cout << "CAN NOT CONVERGE\n";
		return;
	}
	generateB(b);
	T eps = sizeof(a[0]) == 4 ? T(1e-6) : T(1e-12);
	int nIter = 200;

	std::cout << "Intel(R) Core(TM) i5-7500 VS NVIDIA GeForce GTX 1050 Ti\n******************************************************************\n";
	for (int i = 0; i < n; i++) x0[i] = 0, x1[i] = 0;
	cpu_time = opencl_jacobi(2, 1, n, a, b, x0, x1, delta, filename, kernelname, 1, eps, nIter);
	std::cout << "cpu time = \t\t\t" << cpu_time.first << ";   \t" << cpu_time.second << '\n';
	for (int i = 0; i < n; i++) x0[i] = 0, x1[i] = 0;
	gpu_time = opencl_jacobi(2, 1, n, a, b, x0, x1, delta, filename, kernelname, 0, eps, nIter);
	std::cout << "gpu time = \t\t\t" << gpu_time.first << ";   \t" << gpu_time.second << '\n';
	for(int j = 1;j < 10;j++){
		for (int i = 0; i < n; i++) x0[i] = 0, x1[i] = 0;
		cpu_gpu_time = opencl_jacobi(2, 1, n, a, b, x0, x1, delta, filename, kernelname, (double)j / 10, eps, nIter);
		std::cout << (double)j / 10 << " * cpu & " << (1.0 - (double)j / 10) << " * gpu time = \t" << cpu_gpu_time.first << ";   \t" << cpu_gpu_time.second << '\n';
	}

	std::cout << "\nIntel(R) Core(TM) i5-7500 VS Intel(R) HD Graphics 630\n******************************************************************\n";
	for (int i = 0; i < n; i++) x0[i] = 0, x1[i] = 0;
	cpu_time = opencl_jacobi(2, 0, n, a, b, x0, x1, delta, filename, kernelname, 1, eps, nIter);
	std::cout << "cpu time = \t\t\t" << cpu_time.first << ";   \t" << cpu_time.second << '\n';
	for (int i = 0; i < n; i++) x0[i] = 0, x1[i] = 0;
	gpu_time = opencl_jacobi(2, 0, n, a, b, x0, x1, delta, filename, kernelname, 0, eps, nIter);
	std::cout << "gpu time = \t\t\t" << gpu_time.first << ";   \t" << gpu_time.second << '\n';
	for (int j = 1; j < 10; j++) {
		for (int i = 0; i < n; i++) x0[i] = 0, x1[i] = 0;
		cpu_gpu_time = opencl_jacobi(2, 0, n, a, b, x0, x1, delta, filename, kernelname, (double)j / 10, eps, nIter);
		std::cout << (double)j / 10 << " * cpu & " << (1.0 - (double)j / 10) << " * gpu time = \t" << cpu_gpu_time.first << ";   \t" << cpu_gpu_time.second << '\n';
	}
	std::cout << '\n';
}

int main() {
	lets_go<float>("FLOAT");
	lets_go<double>("DOUBLE");
}

