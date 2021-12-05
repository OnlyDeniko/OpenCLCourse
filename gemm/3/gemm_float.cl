__kernel void gemmFloat( int n, int m, int k, __global const float* a, __global const float* b, __global float* c) {
	int i = get_global_id(1);
	int j = get_global_id(0);
	float res = 0;
	for (int kk = 0; kk < m;kk++) {
		res += a[i * m + kk] * b[kk * k + j];
	}
	c[i * k + j] = res;
}