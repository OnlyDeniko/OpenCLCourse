__kernel void jacobiDouble(__global const double* a,
						   __global const double* b,
						   __global double* x0,
						   __global double* x1,
						   __global double* delta) {
	const int j = get_global_id(0);
	const int n = get_global_size(0);
	double ans = 0;

	for (int i = 0; i < n; i++) {
		ans += a[j * n + i] * x0[i] * (double)(i != j);
	}

	x1[j] = (b[j] - ans) / a[j * n + j];
	delta[j] = (x1[j] - x0[j]) / x0[j];
}