// vtk_writer.h —— 接口 4：数值解写 VTK legacy ASCII，ParaView 直接打开
//
// 文件五节：header / DATASET UNSTRUCTURED_GRID
//   POINTS（结点坐标，z=0）→ CELLS + CELL_TYPES（5 = VTK_TRIANGLE）
//   → POINT_DATA：u 与 |u − u_exact| 两个节点标量场
// 纪律：输出模块独立，内核（网格/装配/求解）不认识 VTK。

#pragma once

#include <cmath>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "Mesh.h"

class VtkWriter
{
public:
    // 不带精确解：只写 u
    static void write(const std::string& path, const Mesh2D& mesh,
                      const std::vector<double>& u)
    {
        writeImpl(path, mesh, u, nullptr);
    }

    // 带精确解：附误差场 |u − u_exact|
    template <typename Func>
    static void write(const std::string& path, const Mesh2D& mesh,
                      const std::vector<double>& u, Func uExact)
    {
        const std::function<double(double, double)> fn = uExact;
        writeImpl(path, mesh, u, &fn);
    }

private:
    // uExact 为空表示不写误差场（类型擦除，避免模板参数推导）
    static void writeImpl(const std::string& path, const Mesh2D& mesh,
                          const std::vector<double>& u,
                          const std::function<double(double, double)>* uExact)
    {
        assert((int)u.size() == mesh.numNodes());
        std::ofstream out(path);
        assert(out.good() && "VTK 输出文件打不开");

        out << "# vtk DataFile Version 3.0\n"
            << "fem2d\n"
            << "ASCII\n"
            << "DATASET UNSTRUCTURED_GRID\n";

        // ---- 结点 ----
        out << "POINTS " << mesh.numNodes() << " float\n";
        for (int k = 0; k < mesh.numNodes(); ++k) {
            const auto& p = mesh.node(k);
            out << p[0] << " " << p[1] << " 0\n";
        }

        // ---- 单元（每行：3 + 三个顶点号）----
        out << "CELLS " << mesh.numElements() << " " << mesh.numElements() * 4 << "\n";
        for (int e = 0; e < mesh.numElements(); ++e) {
            const auto& tri = mesh.element(e);
            out << "3 " << tri[0] << " " << tri[1] << " " << tri[2] << "\n";
        }
        out << "CELL_TYPES " << mesh.numElements() << "\n";
        for (int e = 0; e < mesh.numElements(); ++e)
            out << "5\n";                              // VTK_TRIANGLE

        // ---- 节点标量场 ----
        out << "POINT_DATA " << mesh.numNodes() << "\n";
        out << "SCALARS u float\nLOOKUP_TABLE default\n";
        for (int k = 0; k < mesh.numNodes(); ++k)
            out << u[k] << "\n";

        if (uExact) {
            out << "SCALARS abs_err float\nLOOKUP_TABLE default\n";
            for (int k = 0; k < mesh.numNodes(); ++k) {
                const auto& p = mesh.node(k);
                out << std::abs(u[k] - (*uExact)(p[0], p[1])) << "\n";
            }
        }
    }
};
