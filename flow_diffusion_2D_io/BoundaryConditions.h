// BoundaryConditions.h —— 接口 2：边界条件入口（INI 文件驱动）
//
// 输入文件示例（改工况不用重编译）：
//   [domain]
//   eps = 1.0
//   f   = 2*pi*pi*sin(pi*x)*sin(pi*y)
//
//   [bc.1]                  ; tag 1 = 左边界（结构化网格的 LEFT）
//   type  = dirichlet
//   value = sin(pi*x)*sin(pi*y)
//
//   [bc.2]                  ; tag 2 = 右边界（Neumann 给 eps*du/dn）
//   type  = neumann
//   value = -pi*sin(pi*y)
//
// 表达式：+ - * / 括号，函数 sin/cos/exp/log/sqrt/abs，常量 pi，变量 x y。
// Neumann 约定：value 给的是 eps*du/dn（弱形式分部积分后的边界通量），
// 与热问题的热通量边界直接对应。纯 Neumann 问题矩阵奇异，不支持。

#pragma once

#include <cassert>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "Mesh.h"
#include "Assembler.h"

// ---------- 表达式求值（递归下降，只支持上面列出的语法）----------
namespace expr
{

inline void skipSpaces(const std::string& s, size_t& p)
{
    while (p < s.size() && std::isspace((unsigned char)s[p])) ++p;
}

inline double parseExpr(const std::string& s, size_t& p, double x, double y);

inline double parsePrimary(const std::string& s, size_t& p, double x, double y)
{
    skipSpaces(s, p);
    if (p >= s.size()) return 0.0;

    if (s[p] == '(') {
        ++p;
        const double v = parseExpr(s, p, x, y);
        skipSpaces(s, p);
        assert(s[p] == ')'); ++p;
        return v;
    }
    if (std::isdigit((unsigned char)s[p]) || s[p] == '.') {
        size_t q = p;
        while (q < s.size() && (std::isdigit((unsigned char)s[q]) || s[q] == '.')) ++q;
        const double v = std::stod(s.substr(p, q - p));
        p = q;
        return v;
    }
    if (std::isalpha((unsigned char)s[p])) {
        size_t q = p;
        while (q < s.size() && std::isalpha((unsigned char)s[q])) ++q;
        const std::string name = s.substr(p, q - p);
        p = q;
        if (name == "x") return x;
        if (name == "y") return y;
        if (name == "pi") return std::acos(-1.0);
        skipSpaces(s, p);
        assert(s[p] == '('); ++p;
        const double arg = parseExpr(s, p, x, y);
        skipSpaces(s, p);
        assert(s[p] == ')'); ++p;
        if (name == "sin")  return std::sin(arg);
        if (name == "cos")  return std::cos(arg);
        if (name == "exp")  return std::exp(arg);
        if (name == "log")  return std::log(arg);
        if (name == "sqrt") return std::sqrt(arg);
        if (name == "abs")  return std::fabs(arg);
        assert(false && "未知函数");
    }
    assert(false && "无法解析的表达式");
    return 0.0;
}

inline double parseUnary(const std::string& s, size_t& p, double x, double y)
{
    skipSpaces(s, p);
    if (p < s.size() && s[p] == '-') { ++p; return -parseUnary(s, p, x, y); }
    if (p < s.size() && s[p] == '+') { ++p; return  parseUnary(s, p, x, y); }
    return parsePrimary(s, p, x, y);
}

inline double parseTerm(const std::string& s, size_t& p, double x, double y)
{
    double v = parseUnary(s, p, x, y);
    for (;;) {
        skipSpaces(s, p);
        if (p < s.size() && s[p] == '*') { ++p; v *= parseUnary(s, p, x, y); }
        else if (p < s.size() && s[p] == '/') { ++p; v /= parseUnary(s, p, x, y); }
        else return v;
    }
}

inline double parseExpr(const std::string& s, size_t& p, double x, double y)
{
    double v = parseTerm(s, p, x, y);
    for (;;) {
        skipSpaces(s, p);
        if (p < s.size() && s[p] == '+') { ++p; v += parseTerm(s, p, x, y); }
        else if (p < s.size() && s[p] == '-') { ++p; v -= parseTerm(s, p, x, y); }
        else return v;
    }
}

// 求值入口
inline double eval(const std::string& s, double x, double y)
{
    size_t p = 0;
    return parseExpr(s, p, x, y);
}

} // namespace expr

// ---------- 工况描述 ----------
struct BCSpec
{
    enum Type { Dirichlet, Neumann };
    Type        type;
    std::string valueExpr;
};

struct ProblemSpec
{
    std::string epsExpr = "1.0";
    std::string bxExpr  = "0.0";
    std::string byExpr  = "0.0";
    std::string cExpr   = "0.0";
    std::string fExpr;
    std::map<int, BCSpec> bc;        // 边界 tag → 条件
};

// ---------- INI 解析（最小实现：key=value，[section]，# 和 ; 注释）----------
class IniParser
{
public:
    static ProblemSpec read(const std::string& path)
    {
        std::ifstream in(path);
        assert(in.good() && "边界条件文件打不开");

        ProblemSpec spec;
        std::string section, line;
        while (std::getline(in, line)) {
            const auto hash = line.find_first_of("#;");
            if (hash != std::string::npos) line.erase(hash);
            std::istringstream ss(line);
            std::string key;
            ss >> std::skipws;
            if (!(ss >> key) || key.empty()) continue;

            if (key.front() == '[' && key.back() == ']') { section = key.substr(1, key.size() - 2); continue; }

            std::string eq, value;
            ss >> eq;
            std::getline(ss, value);
            const auto b = value.find_first_not_of(" \t");
            value = (b == std::string::npos) ? "" : value.substr(b);

            if (section == "domain") {
                if      (key == "eps") spec.epsExpr = value;
                else if (key == "bx")  spec.bxExpr  = value;
                else if (key == "by")  spec.byExpr  = value;
                else if (key == "c")   spec.cExpr   = value;
                else if (key == "f")   spec.fExpr   = value;
            }
            else if (section.rfind("bc.", 0) == 0) {
                const int tag = std::stoi(section.substr(3));
                auto& bc = spec.bc[tag];
                if (key == "type") {
                    assert(value == "dirichlet" || value == "neumann");
                    bc.type = (value == "dirichlet") ? BCSpec::Dirichlet : BCSpec::Neumann;
                }
                else if (key == "value") bc.valueExpr = value;
            }
        }
        assert(!spec.fExpr.empty() && "INI 缺 [domain] f");
        assert(!spec.bc.empty()     && "INI 没有 [bc.N] 段");
        return spec;
    }
};

// ---------- 自然边界的"配置层入口" ----------
// 物理装配在 Assembler2D::applyNeumann（内核），这里只做翻译：
// 把 INI 里每条 neumann 边界翻成一次内核调用（位测试谓词 + 表达式求值）。
// 角点多 tag 的限制同内核：混边界请用位不重叠的 tag。
inline void applyNeumann(std::vector<double>& rhs, const Mesh2D& mesh,
                         const ProblemSpec& spec)
{
    for (const auto& kv : spec.bc) {
        if (kv.second.type != BCSpec::Neumann) continue;
        assert(!kv.second.valueExpr.empty());
        const int tag = kv.first;
        auto isNatural = [tag](int flag){ return (flag & tag) != 0; };
        auto g = [&kv](double x, double y){ return expr::eval(kv.second.valueExpr, x, y); };
        Assembler2D::applyNeumann(rhs, mesh, isNatural, g);
    }
}
