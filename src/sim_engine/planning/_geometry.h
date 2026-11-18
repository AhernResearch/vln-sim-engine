// Shared stable-ABI geometry primitives and bounded task-owned query caches.
#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sim_engine_native {
struct PythonError {};
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* value) : p(value) { if (!p) throw PythonError{}; }
    ~Ref() { Py_DECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
};
using Point = std::array<double, 3>;
inline bool point(PyObject* value, Point& p) {
    if (!PySequence_Check(value)) return false;
    const auto size = PySequence_Size(value);
    if (size < 0) throw PythonError{};
    if (size != 3) return false;
    for (int i = 0; i < 3; ++i) {
        Ref v(PySequence_GetItem(value, i)); p[i] = PyFloat_AsDouble(v.p);
        if (PyErr_Occurred()) throw PythonError{};
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}
inline PyObject* tuple(const Point& p) { return Py_BuildValue("(ddd)", p[0], p[1], p[2]); }
inline Point difference(const Point& a, const Point& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
inline double squared(const Point& p) { return (p[0]*p[0]+p[1]*p[1])+p[2]*p[2]; }
inline double distance(const Point& a, const Point& b) { return std::sqrt(squared(difference(a,b))); }
inline void checkpoint(PyObject* callback, int query=0, int expansion=0) {
    Ref args(Py_BuildValue("(ii)",query,expansion)); Ref result(PyObject_CallObject(callback,args.p));
    if (PyErr_CheckSignals() < 0) throw PythonError{};
}
template<class F> PyObject* guarded(F fn) {
    try { return fn(); }
    catch (const PythonError&) { return nullptr; }
    catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::invalid_argument& e) { PyErr_SetString(PyExc_ValueError,e.what());return nullptr; }
    catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError,e.what());return nullptr; }
}
using Bits = std::array<std::uint32_t,3>;
inline std::size_t mix(std::size_t h, std::size_t v) { return h^(v+0x9e3779b9+(h<<6)+(h>>2)); }
struct BitsHash {
    std::size_t operator()(const Bits& k) const { return mix(mix(k[0],k[1]),k[2]); }
};
inline Bits quantized(const Point& input, Point& output) {
    Bits key;
    for (int i=0;i<3;++i) {
        const float v=static_cast<float>(input[i]); output[i]=v;
        std::memcpy(&key[i],&v,sizeof(v));
    }
    return key;
}
struct ClearanceKey {
    Bits point; std::uint64_t radius;
    bool operator==(const ClearanceKey& o) const { return point==o.point && radius==o.radius; }
};
struct ClearanceHash {
    std::size_t operator()(const ClearanceKey& k) const { return mix(BitsHash{}(k.point),std::hash<std::uint64_t>{}(k.radius)); }
};
template<class K,class V,class Hash> struct Memo {
    using List=std::list<std::pair<K,V>>;
    std::size_t capacity,hits=0,misses=0,evictions=0;
    List order;
    std::unordered_map<K,typename List::iterator,Hash> index;
    explicit Memo(std::size_t n):capacity(n) {}
    const V* get(const K& key) {
        auto i=index.find(key);
        if (i==index.end()) { ++misses;return nullptr; }
        ++hits; order.splice(order.end(),order,i->second);return &i->second->second;
    }
    void put(const K& key,const V& value) {
        if (!capacity) return;
        auto found=index.find(key);
        if (found!=index.end()) { found->second->second=value;order.splice(order.end(),order,found->second);return; }
        order.emplace_back(key,value);
        try { index.emplace(key,std::prev(order.end())); }
        catch (...) { order.pop_back();throw; }
        if (order.size()>capacity) { index.erase(order.front().first);order.pop_front();++evictions; }
    }
    void clear() { index.clear();order.clear();hits=misses=evictions=0; }
    PyObject* stats() const {
        return Py_BuildValue("{sK,sK,sK,sK,sK}","entries",static_cast<unsigned long long>(order.size()),
            "capacity",static_cast<unsigned long long>(capacity),"hits",static_cast<unsigned long long>(hits),
            "misses",static_cast<unsigned long long>(misses),"evictions",static_cast<unsigned long long>(evictions));
    }
};
struct Snap { Point p; bool valid; };
struct Queries {
    Memo<Bits,Snap,BitsHash> snaps;
    Memo<ClearanceKey,double,ClearanceHash> clearances;
    Memo<Bits,long,BitsHash> islands;
    explicit Queries(std::size_t n):snaps(n),clearances(n),islands(n) {}
    void clear() { snaps.clear();clearances.clear();islands.clear(); }
};
Queries& queries(PyObject* capsule);
Snap snap(Queries& cache,const Point& input,PyObject* callback);
double clearance(Queries& cache,const Point& input,double radius,PyObject* callback);
long island(Queries& cache,const Point& input,PyObject* callback);
PyObject* query_cache(PyObject*,PyObject*);
PyObject* query_clear(PyObject*,PyObject*);
PyObject* query_stats(PyObject*,PyObject*);
PyObject* snap_many(PyObject*,PyObject*);
PyObject* clearance_many(PyObject*,PyObject*);
PyObject* island_many(PyObject*,PyObject*);
PyObject* clearance_one(PyObject*,PyObject*);
PyObject* swept_edge(PyObject*,PyObject*);
PyObject* fair_path(PyObject*,PyObject*);
} // namespace sim_engine_native
