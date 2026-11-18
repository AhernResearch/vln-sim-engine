// Exact-order implicit corridor A*. NavMesh and safety checks remain callbacks.
// CPython stable ABI: no NumPy, Habitat, Boost or simulator linkage required.
#include "_geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using namespace sim_engine_native;
std::int64_t rounded(double v) {
    if (!std::isfinite(v) || v <= -9e18 || v >= 9e18)
        throw std::invalid_argument("native A* floor key exceeds integer range");
    // Python round(x): nearest integer, ties to even, independent of fenv.
    const double lower = std::floor(v), fraction = v-lower;
    const auto i = static_cast<std::int64_t>(lower);
    return i + (fraction > .5 || (fraction == .5 && i % 2 != 0));
}
struct Key {
    std::int64_t x, z, floor;
    bool operator==(const Key& other) const {
        return x == other.x && z == other.z && floor == other.floor;
    }
};
struct Hash {
    std::size_t operator()(const Key& k) const {
        auto h = std::hash<std::int64_t>{}(k.x);
        for (auto v : {k.z, k.floor}) h ^= std::hash<std::int64_t>{}(v) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};
struct Node { Point p; double cost; Key parent; bool has_parent; };
struct Entry { double rank; std::uint64_t serial; double cost; Key key; };
struct Later {
    bool operator()(const Entry& a, const Entry& b) const {
        return a.rank > b.rank || (a.rank == b.rank && a.serial > b.serial);
    }
};
struct Segment { Point a, delta; double denominator; };
struct Search {
    Point start, goal;
    double grid, height, floor_bin, radius;
    Py_ssize_t limit;
    PyObject *snap, *edge, *check;
    Queries* query_memo=nullptr;
    std::vector<Segment> segments;
    void checkpoint(int queries, int expansions) {
        Ref args(Py_BuildValue("(ii)", queries, expansions));
        Ref result(PyObject_CallObject(check, args.p));
        if (PyErr_CheckSignals() < 0) throw PythonError{};
    }
    bool inside(const Point& p) {
        checkpoint(1, 0);
        for (const auto& s : segments) {
            const auto offset = difference(p, s.a);
            const auto& d = s.delta;
            double t = ((offset[0]*d[0] + offset[1]*d[1]) + offset[2]*d[2]) / s.denominator;
            t = std::max(0., std::min(1., t));
            const Point projection{s.a[0]+t*d[0], s.a[1]+t*d[1], s.a[2]+t*d[2]};
            if (distance(p, projection) <= radius) return true;
        }
        return false;
    }
    bool snapped(const Point& p, Point& output) {
        if (query_memo) {
            const auto result=sim_engine_native::snap(*query_memo,p,snap);
            output=result.p;return result.valid;
        }
        Point quantized{static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])};
        Ref input(tuple(quantized));
        Ref args(PyTuple_Pack(1, input.p));
        Ref result(PyObject_CallObject(snap, args.p));
        return point(result.p, output);
    }
    bool edge_clear(const Point& a, const Point& b) {
        Ref first(tuple(a)), second(tuple(b));
        Ref args(PyTuple_Pack(2, first.p, second.p));
        Ref result(PyObject_CallObject(edge, args.p));
        const int valid = PyObject_IsTrue(result.p);
        if (valid < 0) throw PythonError{};
        return valid;
    }
    PyObject* run() {
        const Key first{0, 0, rounded(start[1]/floor_bin)};
        std::unordered_map<Key, Node, Hash> nodes;
        nodes.emplace(first, Node{start, 0., {}, false});
        std::priority_queue<Entry, std::vector<Entry>, Later> queue;
        std::uint64_t serial = 0;
        queue.push({distance(goal, start), serial++, 0., first});
        Py_ssize_t expanded = 0;
        while (!queue.empty()) {
            checkpoint(0, 1);
            const auto entry = queue.top(); queue.pop();
            const auto node = nodes.at(entry.key); // Copy: insertions may rehash.
            if (entry.cost != node.cost) continue;
            if (++expanded > limit)
                throw std::invalid_argument("clearance search expansion budget exhausted");
            const auto current = node.p;
            if (distance(current, goal) <= grid*1.5 && edge_clear(current, goal)) {
                std::vector<Point> path{goal, current};
                auto key = entry.key;
                while (nodes.at(key).has_parent) {
                    key = nodes.at(key).parent;
                    path.push_back(nodes.at(key).p);
                }
                Ref result(PyList_New(0));
                for (auto p = path.rbegin(); p != path.rend(); ++p) {
                    Ref row(tuple(*p));
                    if (PyList_Append(result.p, row.p) < 0) throw PythonError{};
                }
                return Py_BuildValue("(On)", result.p, expanded);
            }
            for (int dx = -1; dx <= 1; ++dx) for (int dz = -1; dz <= 1; ++dz) {
                if (!dx && !dz) continue;
                const auto ix = entry.key.x+dx, iz = entry.key.z+dz;
                const Point candidate{start[0]+ix*grid, current[1], start[2]+iz*grid};
                if (!inside(candidate)) continue;
                Point p;
                if (!snapped(candidate, p)) continue;
                const auto offset = difference(p, candidate);
                if (std::sqrt(offset[0]*offset[0]+offset[2]*offset[2]) > 1e-4) continue;
                if (std::abs(p[1]-current[1]) > height || !inside(p)) continue;
                const Key key{ix, iz, rounded(p[1]/floor_bin)};
                const double cost = entry.cost+distance(p, current);
                const auto found = nodes.find(key);
                if (found != nodes.end() && cost >= found->second.cost) continue;
                if (!edge_clear(current, p)) continue;
                nodes.insert_or_assign(key, Node{p, cost, entry.key, true});
                queue.push({cost+distance(goal, p), serial++, cost, key});
            }
        }
        throw std::invalid_argument("no route with requested clearance inside the reference corridor");
    }
};

PyObject* search(PyObject*, PyObject* args) {
    try {
        PyObject *a, *b, *seed,*arena=Py_None;
        Search s;
        if (!PyArg_ParseTuple(args, "OOOddddnOOO|O", &a, &b, &seed, &s.grid, &s.height,
                              &s.floor_bin, &s.radius, &s.limit, &s.snap, &s.edge, &s.check,&arena)) return nullptr;
        if (arena!=Py_None) s.query_memo=&queries(arena);
        if (!point(a, s.start) || !point(b, s.goal))
            throw std::invalid_argument("finite XYZ endpoints required");
        for (auto v : {s.grid, s.height, s.floor_bin, s.radius})
            if (!std::isfinite(v) || v <= 0) throw std::invalid_argument("positive finite search profile required");
        if (s.limit < 1 || s.limit > 1000000) throw std::invalid_argument("invalid native expansion limit");
        const auto count = PySequence_Size(seed);
        if (count < 0) throw PythonError{};
        if (count < 2) throw std::invalid_argument("at least two reference points required");
        Point previous;
        for (Py_ssize_t i = 0; i < count; ++i) {
            if (i % 32 == 0) s.checkpoint(0, 0);
            Ref row(PySequence_GetItem(seed, i));
            Point p;
            if (!point(row.p, p)) throw std::invalid_argument("finite XYZ reference points required");
            if (i) {
                const auto d = difference(p, previous);
                s.segments.push_back({previous, d, std::max(squared(d), 1e-12)});
            }
            previous = p;
        }
        return s.run();
    } catch (const PythonError&) {
        return nullptr;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what()); return nullptr;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what()); return nullptr;
    }
}
PyMethodDef methods[] = {
    {"search",search,METH_VARARGS,"Bounded implicit corridor A* with stable tie order."},
    {"query_cache",query_cache,METH_VARARGS,"Create a task-owned bounded geometry cache."},
    {"query_clear",query_clear,METH_VARARGS,"Release cached task queries."},
    {"query_stats",query_stats,METH_VARARGS,"Query memo counters."},
    {"snap_many",snap_many,METH_VARARGS,"Ordered bounded batch of exact snap queries."},
    {"clearance_many",clearance_many,METH_VARARGS,"Ordered clearance queries and logical charges."},
    {"clearance_one",clearance_one,METH_VARARGS,"One exact cached clearance query."},
    {"island_many",island_many,METH_VARARGS,"Ordered island queries."},
    {"swept_edge",swept_edge,METH_VARARGS,"Sequential swept segment checks."},
    {"fair_path",fair_path,METH_VARARGS,"Sequential in-place path fairing."},
    {nullptr,nullptr,0,nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_astar", nullptr, -1, methods, nullptr, nullptr, nullptr, nullptr};
} // namespace
PyMODINIT_FUNC PyInit__astar() {
    PyObject* value=PyModule_Create(&module);
    if (value && PyModule_AddIntConstant(value,"KERNEL_API",2)<0) { Py_DECREF(value);return nullptr; }
    return value;
}
