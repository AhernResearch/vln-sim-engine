#include "_geometry.h"
#include <memory>

namespace sim_engine_native {
namespace {
constexpr const char* name="sim_engine.navigation.NavMeshQueries.v1";
void destroy(PyObject* capsule) {
    auto* value=static_cast<Queries*>(PyCapsule_GetPointer(capsule,name));
    if (value) delete value; else PyErr_Clear();
}
}
Queries& queries(PyObject* capsule) {
    auto* value=static_cast<Queries*>(PyCapsule_GetPointer(capsule,name));
    if (!value) throw PythonError{};
    return *value;
}
Snap snap(Queries& cache,const Point& input,PyObject* callback) {
    Point p;const auto key=quantized(input,p);
    if (const auto* v=cache.snaps.get(key)) return *v;
    Ref row(tuple(p));Ref args(PyTuple_Pack(1,row.p));Ref value(PyObject_CallObject(callback,args.p));
    Snap result{};result.valid=point(value.p,result.p);
    if (!result.valid) result.p.fill(std::numeric_limits<double>::quiet_NaN());
    cache.snaps.put(key,result);return result;
}
double clearance(Queries& cache,const Point& input,double radius,PyObject* callback) {
    if (!std::isfinite(radius) || radius<=0) throw std::invalid_argument("invalid clearance radius");
    Point p;ClearanceKey key{quantized(input,p),0};std::memcpy(&key.radius,&radius,sizeof(radius));
    if (const auto* v=cache.clearances.get(key)) return *v;
    Ref row(tuple(p));Ref args(Py_BuildValue("(Od)",row.p,radius));Ref value(PyObject_CallObject(callback,args.p));
    const double result=PyFloat_AsDouble(value.p);
    if (PyErr_Occurred()) throw PythonError{};
    if (!std::isfinite(result) || result<0) throw std::invalid_argument("invalid navmesh clearance");
    cache.clearances.put(key,result);return result;
}
long island(Queries& cache,const Point& input,PyObject* callback) {
    Point p;const auto key=quantized(input,p);
    if (const auto* v=cache.islands.get(key)) return *v;
    Ref row(tuple(p));Ref args(PyTuple_Pack(1,row.p));Ref value(PyObject_CallObject(callback,args.p));
    const long result=PyLong_AsLong(value.p);
    if (PyErr_Occurred()) throw PythonError{};
    cache.islands.put(key,result);return result;
}
PyObject* query_cache(PyObject*,PyObject* args) {
    return guarded([&]()->PyObject* {
        Py_ssize_t capacity;
        if (!PyArg_ParseTuple(args,"n",&capacity)) return nullptr;
        if (capacity<0 || capacity>1048576) throw std::invalid_argument("invalid native query capacity");
        auto value=std::make_unique<Queries>(static_cast<std::size_t>(capacity));
        PyObject* result=PyCapsule_New(value.get(),name,destroy);
        if (!result) throw PythonError{};
        value.release();return result;
    });
}
PyObject* query_clear(PyObject*,PyObject* args) {
    return guarded([&]()->PyObject* {
        PyObject* capsule;if (!PyArg_ParseTuple(args,"O",&capsule)) return nullptr;
        queries(capsule).clear();Py_RETURN_NONE;
    });
}
PyObject* query_stats(PyObject*,PyObject* args) {
    return guarded([&]()->PyObject* {
        PyObject* capsule;if (!PyArg_ParseTuple(args,"O",&capsule)) return nullptr;
        auto& q=queries(capsule);Ref a(q.snaps.stats()),b(q.clearances.stats()),c(q.islands.stats());
        return Py_BuildValue("{sO,sO,sO}","snap",a.p,"clearance",b.p,"island",c.p);
    });
}
namespace {
template<class F> PyObject* batch(PyObject* capsule,PyObject* points,PyObject* check,int charge,F fn) {
    auto& q=queries(capsule);const auto size=PySequence_Size(points);
    if (size<0) throw PythonError{};
    if (size>4096) throw std::invalid_argument("native query batch exceeds limit");
    Ref output(PyList_New(0));
    for (Py_ssize_t i=0;i<size;++i) {
        checkpoint(check,charge);Ref row(PySequence_GetItem(points,i));Point p;
        if (!point(row.p,p)) throw std::invalid_argument("finite XYZ query points required");
        Ref value(fn(q,p));if (PyList_Append(output.p,value.p)<0) throw PythonError{};
    }
    Py_INCREF(output.p);return output.p;
}
}
PyObject* snap_many(PyObject*,PyObject* args) {
    return guarded([&]()->PyObject* {
        PyObject *capsule,*points,*fn,*check;
        if (!PyArg_ParseTuple(args,"OOOO",&capsule,&points,&fn,&check)) return nullptr;
        return batch(capsule,points,check,0,[&](Queries& q,const Point& p){return tuple(snap(q,p,fn).p);});
    });
}
PyObject* clearance_many(PyObject*,PyObject* args) {
    return guarded([&]()->PyObject* {
        PyObject *capsule,*points,*fn,*check;double radius;
        if (!PyArg_ParseTuple(args,"OOdOO",&capsule,&points,&radius,&fn,&check)) return nullptr;
        return batch(capsule,points,check,1,[&](Queries& q,const Point& p){return PyFloat_FromDouble(clearance(q,p,radius,fn));});
    });
}
PyObject* island_many(PyObject*,PyObject* args) {
    return guarded([&]()->PyObject* {
        PyObject *capsule,*points,*fn,*check;
        if (!PyArg_ParseTuple(args,"OOOO",&capsule,&points,&fn,&check)) return nullptr;
        return batch(capsule,points,check,0,[&](Queries& q,const Point& p){return PyLong_FromLong(island(q,p,fn));});
    });
}
PyObject* clearance_one(PyObject*,PyObject* args) {
    return guarded([&]()->PyObject* {
        PyObject *capsule,*input,*fn,*check;double radius;
        if (!PyArg_ParseTuple(args,"OOdOO",&capsule,&input,&radius,&fn,&check)) return nullptr;
        checkpoint(check,1);Point p;
        if (!point(input,p)) throw std::invalid_argument("finite XYZ query point required");
        return PyFloat_FromDouble(clearance(queries(capsule),p,radius,fn));
    });
}
} // namespace sim_engine_native
