__kernel void jacobiFloat(__global const float* a,
						  __global const float* b,
						  __global float* x0,
						  __global float* x1,
						  __global float* delta) {
	const int j = get_global_id(0);
	const int n = get_global_size(0);
	float ans = 0;

	for(int i = 0;i < n;i++){
		ans += a[j * n + i] * x0[i] * (float)(i != j);
	}

	x1[j] = (b[j] - ans) / a[j * n + j];
	delta[j] = (x1[j] - x0[j]) / x0[j];
}