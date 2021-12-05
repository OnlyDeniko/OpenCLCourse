#include<omp.h>

#define BLOCK_SIZE 16

template<typename T>
double omp_gemm(int n, int m, int k, const T const * a, const T const * b, T* c){
	double time = omp_get_wtime();
	int i, j, kk;
	T res;
#pragma omp parallel for shared(n, m, k, a, b, c) private(i, j, kk, res)
	for(i = 0;i < n;i++){
		for(j = 0;j < k;j++){
			res = 0;
			for(kk = 0;kk < m;kk++){
				res += a[i * m + kk] * b[kk * k + j];
			}
			c[i * k + j] = res;
		}
	}
	time = omp_get_wtime() - time;
	return time;
}

template<typename T>
double omp_gemm_block_2(int n, int m, int k, const T const* a, const T const* b, T* c) {
	double time = omp_get_wtime();
	int block_1, block_2, i, j, kk;
	T res;
	int nBlocks = n / BLOCK_SIZE;
	int kBlocks = k / BLOCK_SIZE;
#pragma omp parallel for shared(n, m, nBlocks, kBlocks, k, a, b, c) private(block_1, block_2, i, j, kk, res)
	for(block_1 = 0;block_1 < nBlocks;block_1++){
		for(block_2 = 0;block_2 < kBlocks;block_2++){
			//[block_1, block_2]      ..... [block_1, block_2 + 15]
			// ...
			//[block_1 + 15, block_2] ..... [block_1 + 15, block_2 + 15]
			int nFinish = std::min(n, BLOCK_SIZE * (block_1 + 1));
			int kFinish = std::min(k, BLOCK_SIZE * (block_2 + 1));
			for(i = block_1 * BLOCK_SIZE;i < nFinish;i++){
				for(j = block_2 * BLOCK_SIZE; j < kFinish;j++) {
					res = 0;
					for(kk = 0;kk < m;kk++){
						res += a[i * m + kk] * b[kk * k + j];
					}
					c[i * k + j] = res;
				}
			}
		}
	}
	time = omp_get_wtime() - time;
	return time;
}

template<typename T>
double omp_gemm_block_1(int n, int m, int k, const T const* a, const T const* b, T* c) {
	double time = omp_get_wtime();
	int block_1, block_2, block_3, i, j, kk;
	int nBlocks = n / BLOCK_SIZE;
	int kBlocks = k / BLOCK_SIZE;
	int mBlocks = m / BLOCK_SIZE;
#pragma omp parallel for shared(nBlocks, kBlocks, mBlocks, a, b, c) private(block_1, block_2, block_3, i, j, kk) collapse(6)
	for (block_1 = 0; block_1 < nBlocks; block_1++) {
		for (block_2 = 0; block_2 < kBlocks; block_2++) {
			for (block_3 = 0;block_3 < mBlocks;block_3++){
				for(i = block_1 * BLOCK_SIZE;i < (block_1 + 1) * BLOCK_SIZE;i++){
					for(j = block_2 * BLOCK_SIZE;j < (block_2 + 1) * BLOCK_SIZE;j++){
						for(kk = block_3 * BLOCK_SIZE;kk < (block_3 + 1) * BLOCK_SIZE;kk++){
							c[i * k + j] += a[i * m + kk] * b[kk * k + j];
						}
					}
				}
			}
		}
	}
	time = omp_get_wtime() - time;
	return time;
}