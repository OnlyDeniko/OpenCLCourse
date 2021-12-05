#define EPS 0.0001

__kernel void jacobiDouble(
		__global const double* a,
		__global const double* b,
		__global double* x0,
		__global double* x1,
		__global double* delta,
		int n,
		int stride) {
	const int j = get_global_id(0);
	double ans = 0;

	for (int i = 0; i < n; i++) {
		ans += a[j * n + i] * x0[i] * (double)(i != (j + stride));
	}

	x1[j + stride] = (b[j + stride] - ans) / a[j * n + j + stride];
	delta[j + stride] = x1[j + stride] - x0[j + stride];
	if (x0[j + stride] > EPS) delta[j + stride] /= x0[j + stride];
}