// gmsh_reader.h —— 接口 1：读 Gmsh .msh（锁定 2.2 ASCII），生成 Mesh2D
//
// 解析三步（.msh 2.2 格式三节）：
//   $PhysicalNames  tag → 名字映射（只用于打印核对）
//   $Nodes          Gmsh 编号从 1 起，内部重映射为 0-based
//   $Elements       类型 2 = 三结点三角形（进域，物理 tag = 面 tag）
//                   类型 1 = 二结点线单元（边界，物理 tag → 两个端点的 flag）
//
// 纪律：本文件是唯一允许碰 .msh 格式的地方，Mesh2D 不认识任何文件格式。
// 混合边界注意：角点同时属于多条物理边时 flag 按位或，混边界谓词可能失配，
// 需要混边界时请用结构化网格，或让角点只挂一条边的 tag。

#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "Mesh.h"

class GmshReader
{
public:
    static Mesh2D read(const std::string& path)
    {
        std::ifstream in(path);
        assert(in.good() && ".msh 文件打不开");

        std::vector<std::array<double, 2>> nodes;
        std::vector<std::array<int, 3>>    elements;
        std::vector<int>                   flags;

        std::map<int, int>  gmsh2local;    // Gmsh 结点号（1 起，可能不连续）→ 0-based
        std::map<int, std::string> physNames;

        std::string line;
        while (std::getline(in, line)) {
            if (line == "$MeshFormat") {
                std::getline(in, line);
                std::istringstream ss(line);
                double version; ss >> version;
                assert(version == 2.2 && "只认 .msh 2.2（Gmsh 4.x 默认 4.1，导出时锁 2.2）");
                std::getline(in, line);          // $EndMeshFormat
            }
            else if (line == "$PhysicalNames") {
                std::getline(in, line);
                const int n = std::stoi(line);
                for (int i = 0; i < n; ++i) {
                    std::getline(in, line);
                    std::istringstream ss(line);
                    int dim, tag; std::string name;
                    ss >> dim >> tag;
                    std::getline(ss, name);
                    physNames[tag] = name;
                }
                std::getline(in, line);          // $EndPhysicalNames
            }
            else if (line == "$Nodes") {
                std::getline(in, line);
                const int n = std::stoi(line);
                nodes.reserve(n);
                for (int i = 0; i < n; ++i) {
                    std::getline(in, line);
                    std::istringstream ss(line);
                    long id; double x, y, z;
                    ss >> id >> x >> y >> z;
                    gmsh2local[id] = (int)nodes.size();
                    nodes.push_back({ x, y });
                }
                flags.assign(nodes.size(), Mesh2D::NONE);
                std::getline(in, line);          // $EndNodes
            }
            else if (line == "$Elements") {
                std::getline(in, line);
                const int n = std::stoi(line);
                for (int i = 0; i < n; ++i) {
                    std::getline(in, line);
                    std::istringstream ss(line);
                    long id; int type, ntags;
                    ss >> id >> type >> ntags;
                    std::vector<int> tags(ntags);
                    for (int t = 0; t < ntags; ++t) ss >> tags[t];
                    const int phys = tags.empty() ? 0 : tags[0];   // 第一个 tag = 物理 tag

                    if (type == 2) {               // 三角形
                        long a, b, c; ss >> a >> b >> c;
                        elements.push_back({ gmsh2local[a], gmsh2local[b], gmsh2local[c] });
                    }
                    else if (type == 1) {          // 线单元：把物理 tag 打到两个端点上
                        long a, b; ss >> a >> b;
                        flags[gmsh2local[a]] |= phys;
                        flags[gmsh2local[b]] |= phys;
                    }
                    // 其余单元类型（四边形等）直接跳过——本求解器只认 P1 三角形
                }
                std::getline(in, line);          // $EndElements
            }
        }

        std::cout << "[GmshReader] " << path
                  << "  nodes=" << nodes.size()
                  << "  triangles=" << elements.size() << "\n";
        for (const auto& kv : physNames)
            std::cout << "[GmshReader]   physical tag " << kv.first
                      << " -> " << kv.second << "\n";

        assert(!nodes.empty() && !elements.empty());
        return Mesh2D(std::move(nodes), std::move(elements), std::move(flags));
    }
};
