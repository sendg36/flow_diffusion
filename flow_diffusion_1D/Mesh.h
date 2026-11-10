#pragma once
#include <vector>
struct Mesh1D
{
	int N;
	double h;
	std::vector<double> x;
	Mesh1D(int n, double a, double b) :N(n), h((b - a) / n), x(n + 1) {
		for (int i = 0; i <= n; ++i) x[i] = a + h * i;
	}


};
