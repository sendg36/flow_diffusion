// Mesh.h —— 二维结构化三角网格
// 矩形域均匀剖分，每格劈两个三角形（统一逆时针）。
// 边界一律走 boundaryFlag 位标记，别处不许比坐标——阶段三换 Gmsh 网格时全靠它。

#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <vector>

class Mesh2D
{
public:
    // 边界标记位（角点可同时属于多条边）
    enum BoundaryFlag {
        NONE   = 0,
        LEFT   = 1,   // x == x0
        RIGHT  = 2,   // x == x1
        BOTTOM = 4,   // y == y0
        TOP    = 8    // y == y1
    };

    // 均匀结构化网格；默认单位正方形
    Mesh2D(int nx, int ny,
           double x0 = 0.0, double y0 = 0.0,
           double x1 = 1.0, double y1 = 1.0)
        : nx_(nx), ny_(ny)
    {
        assert(nx >= 1 && ny >= 1);

        const int npx = nx_ + 1;
        const int npy = ny_ + 1;
        numNodes_    = npx * npy;
        numElements_ = 2 * nx_ * ny_;

        nodes_.resize(numNodes_);
        boundaryFlag_.assign(numNodes_, NONE);
        elements_.resize(numElements_);

        const double hx = (x1 - x0) / nx_;
        const double hy = (y1 - y0) / ny_;

        // 结点坐标 + 边界标记
        for (int j = 0; j < npy; ++j)
            for (int i = 0; i < npx; ++i) {
                const int k = nodeIndex(i, j);
                nodes_[k] = { x0 + i * hx, y0 + j * hy };

                int flag = NONE;
                if (i == 0)      flag |= LEFT;
                if (i == nx_)    flag |= RIGHT;
                if (j == 0)      flag |= BOTTOM;
                if (j == ny_)    flag |= TOP;
                boundaryFlag_[k] = flag;
            }

        // 单元连通表：方格 (i,j) 劈成 (n00,n10,n11) 和 (n00,n11,n01)
        int e = 0;
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                const int n00 = nodeIndex(i,     j    );
                const int n10 = nodeIndex(i + 1, j    );
                const int n01 = nodeIndex(i,     j + 1);
                const int n11 = nodeIndex(i + 1, j + 1);

                elements_[e] = makeCCW({ n00, n10, n11 }); ++e;
                elements_[e] = makeCCW({ n00, n11, n01 }); ++e;
            }

        assert(e == numElements_);

        // 手算核对：结点数 / 单元数
        std::cout << "[Mesh2D] nx=" << nx_ << " ny=" << ny_
                  << "  domain=[" << x0 << "," << x1 << "]x["
                  << y0 << "," << y1 << "]\n"
                  << "[Mesh2D] numNodes    = " << numNodes_
                  << "  (expect (nx+1)*(ny+1) = " << npx * npy << ")\n"
                  << "[Mesh2D] numElements = " << numElements_
                  << "  (expect 2*nx*ny = " << 2 * nx_ * ny_ << ")\n";
        assert(numNodes_    == npx * npy);
        assert(numElements_ == 2 * nx_ * ny_);
    }

    int  numNodes()    const { return numNodes_; }
    int  numElements() const { return numElements_; }
    int  nx() const { return nx_; }
    int  ny() const { return ny_; }

    const std::array<double, 2>& node(int k) const { return nodes_[k]; }
    const std::array<int, 3>&    element(int e) const { return elements_[e]; }

    int  boundaryFlag(int k) const { return boundaryFlag_[k]; }
    bool isBoundary(int k)   const { return boundaryFlag_[k] != NONE; }

private:
    // 编号规则只许出现在这里
    int nodeIndex(int i, int j) const { return i + j * (nx_ + 1); }

    static double signedArea2(const std::array<double, 2>& a,
                              const std::array<double, 2>& b,
                              const std::array<double, 2>& c)
    {
        return (b[0] - a[0]) * (c[1] - a[1])
             - (b[1] - a[1]) * (c[0] - a[0]);
    }

    std::array<int, 3> makeCCW(std::array<int, 3> tri) const
    {
        if (signedArea2(nodes_[tri[0]], nodes_[tri[1]], nodes_[tri[2]]) < 0.0)
            std::swap(tri[1], tri[2]);
        return tri;
    }

    int nx_, ny_;
    int numNodes_, numElements_;
    std::vector<std::array<double, 2>> nodes_;
    std::vector<std::array<int, 3>>    elements_;
    std::vector<int>                   boundaryFlag_;
};
