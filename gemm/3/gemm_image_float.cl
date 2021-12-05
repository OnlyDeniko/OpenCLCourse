#define BLOCK_SIZE 16

__kernel void gemmFloatImage(int n, int m, int k, __read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c) {
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
		int2 indexA = { block_col, global_row };
		int2 indexB = { global_col, block_row };
		A[local_row][local_col] = read_imagef(a, indexA).x;
		B[local_row][local_col] = read_imagef(b, indexB).x;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int i = 0; i < BLOCK_SIZE; i++) {
			res += A[local_row][i] * B[i][local_col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	int2 index = { global_col, global_row };
	write_imagef(c, index, res);
}