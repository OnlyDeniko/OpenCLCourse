#define BLOCK_SIZE 16

__kernel void gemmFloatBlock(int n, int m, int k, __global const float* a, __global const float* b, __global float* c) {
	__local float A[BLOCK_SIZE][BLOCK_SIZE];
	__local float B[BLOCK_SIZE][BLOCK_SIZE];
	
	int local_row = get_local_id(1);
	int local_col = get_local_id(0);

	int global_row = get_global_id(1);
	int global_col = get_global_id(0);

	float res = 0;
	int nBlocks = m / BLOCK_SIZE;
	for (int iBlock = 0; iBlock < nBlocks; iBlock++) {
		int block_col = iBlock * BLOCK_SIZE + local_col;
		int block_row = iBlock * BLOCK_SIZE + local_row;
		A[local_row][local_col] = a[global_row * m + block_col];
		B[local_row][local_col] = b[block_row * k + global_col];
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int i = 0;i < BLOCK_SIZE;i++){
			res += A[local_row][i] * B[i][local_col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	c[global_row * k + global_col] = res;
}