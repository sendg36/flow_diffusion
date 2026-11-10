// Post.h —— 后处理：L2 误差
// 逐单元三点中点求积。中点上 uh = 对边两结点值的平均，跟求积公式同阶，
// 误差表里的 rate 比较干净。

#pragma once

#include <cmath>
#include <vector>
#include "Mesh.h"

class Post2D
{
public:
    template <typename Func>
    static double l2Error(const Mesh2D& mesh, const std::vector<double>& uh, Func uExact)
    {
        double err2 = 0.0;
        for (int e = 0; e < mesh.numElements(); ++e) {
            const auto& tri = mesh.element(e);
            const auto& P0 = mesh.node(tri[0]);
            const auto& P1 = mesh.node(tri[1]);
            const auto& P2 = mesh.node(tri[2]);
            const double area2 = (P1[0]-P0[0])*(P2[1]-P0[1])
                               - (P2[0]-P0[0])*(P1[1]-P0[1]);
            const double area = 0.5 * area2;

            // 三个中点：m 对向顶点 m
            for (int m = 0; m < 3; ++m) {
                const int a = (m + 1) % 3, b = (m + 2) % 3;
                const auto& A = mesh.node(tri[a]);
                const auto& B = mesh.node(tri[b]);
                const double px = 0.5 * (A[0] + B[0]);
                const double py = 0.5 * (A[1] + B[1]);
                const double uhMid = 0.5 * (uh[tri[a]] + uh[tri[b]]);
                const double d = uExact(px, py) - uhMid;
                err2 += area / 3.0 * d * d;
            }
        }
        return std::sqrt(err2);
    }
};
