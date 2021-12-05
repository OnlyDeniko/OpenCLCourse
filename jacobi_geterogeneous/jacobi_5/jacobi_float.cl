#define EPS 0.0001

__kernel void jacobiFloat(
		__global const float* a,
		__global const float* b,
		__global float* x0,
		__global float* x1,
		__global float* delta,
		int n,
		int stride) {
	const int j = get_global_id(0);
	float ans = 0;

	for(int i = 0;i < n;i++){
		ans += a[j * n + i] * x0[i] * (float)(i != (j + stride));
	}

	x1[j + stride] = (b[j + stride] - ans) / a[j * n + j + stride];
	delta[j + stride] = x1[j + stride] - x0[j + stride];
	if (x0[j + stride] > EPS) delta[j + stride] /= x0[j + stride];
}