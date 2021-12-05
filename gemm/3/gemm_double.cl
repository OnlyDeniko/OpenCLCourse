__kernel void gemmDouble( int n, int m, int k, __global const double* a, __global const double* b, __global double* c) {
	int i = get_global_id(1);
	int j = get_global_id(0);
	if (i < n && j < k) {
		double res = 0;
		for (int kk = 0; kk < m; kk++) {
			res += a[i * m + kk] * b[kk * k + j];
		}
		c[i * k + j] = res;
	}
}