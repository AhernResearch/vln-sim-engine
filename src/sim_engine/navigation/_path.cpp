#include "_geometry.h"

namespace sim_engine_native {
namespace {
bool beyond(const Point& delta,double threshold,bool horizontal,PyObject* compare) {
    const double norm=horizontal?std::sqrt(delta[0]*delta[0]+delta[2]*delta[2]):std::sqrt(squared(delta));
    // Preserve NumPy's decision at float boundaries; callbacks are rare here.
    if (std::abs(norm-threshold)<=1e-12) {
        Ref row(horizontal?Py_BuildValue("(dd)",delta[0],delta[2]):tuple(delta));
        Ref args(Py_BuildValue("(Od)",row.p,threshold));Ref result(PyObject_CallObject(compare,args.p));
        const int value=PyObject_IsTrue(result.p);if (value<0) throw PythonError{};return value;
    }
    return norm>threshold;
}
bool edge(PyObject* callback,const Point& a,const Point& b) {
    Ref x(tuple(a)),y(tuple(b));Ref args(PyTuple_Pack(2,x.p,y.p));Ref value(PyObject_CallObject(callback,args.p));
    int result=PyObject_IsTrue(value.p);if (result<0) throw PythonError{};return result;
}
}
PyObject* swept_edge(PyObject*,PyObject* args) {
    return guarded([&]()->PyObject* {
        PyObject *first,*last,*capsule,*snap_fn,*clear_fn,*step_fn,*check,*compare;
        Py_ssize_t count;double height,minimum,radius;
        if (!PyArg_ParseTuple(args,"OOndddOOOOOO",&first,&last,&count,&height,&minimum,&radius,
                              &capsule,&snap_fn,&clear_fn,&step_fn,&check,&compare)) return nullptr;
        Point a,b;if (!point(first,a)||!point(last,b)) throw std::invalid_argument("finite edge required");
        if (count<1 || count>1000000 || !std::isfinite(height) || height<=0 ||
            !std::isfinite(minimum) || minimum<=0) throw std::invalid_argument("invalid swept edge profile");
        auto& q=queries(capsule);Point previous=a;Py_ssize_t charged=0;
        const auto delta=difference(b,a);
        for (Py_ssize_t i=0;i<=count;++i) {
            checkpoint(check);
            const Point p=i==0?a:Point{a[0]+delta[0]*i/count,a[1]+delta[1]*i/count,a[2]+delta[2]*i/count};
            const auto s=snap(q,p,snap_fn);
            if (!s.valid || beyond(difference(s.p,p),1e-4,true,compare) || std::abs(s.p[1]-p[1])>height)
                return Py_BuildValue("(On)",Py_False,charged);
            ++charged;checkpoint(check,1);
            if (clearance(q,s.p,radius,clear_fn)+1e-6<minimum) return Py_BuildValue("(On)",Py_False,charged);
            Point from,to;quantized(previous,from);quantized(s.p,to);
            Ref x(tuple(from)),y(tuple(to));Ref call(PyTuple_Pack(2,x.p,y.p));Ref result(PyObject_CallObject(step_fn,call.p));
            Point reached;
            if (!point(result.p,reached) || beyond(difference(reached,s.p),1e-3,false,compare))
                return Py_BuildValue("(On)",Py_False,charged);
            previous=s.p;
        }
        return Py_BuildValue("(On)",Py_True,charged);
    });
}
PyObject* fair_path(PyObject*,PyObject* args) {
    return guarded([&]()->PyObject* {
        PyObject *input,*capsule,*snap_fn,*inside,*edge_fn,*check,*compare;Py_ssize_t passes;double height;
        if (!PyArg_ParseTuple(args,"OndOOOOOO",&input,&passes,&height,&capsule,&snap_fn,&inside,&edge_fn,&check,&compare)) return nullptr;
        auto& q=queries(capsule);auto size=PySequence_Size(input);if (size<0) throw PythonError{};
        if (size<2 || size>1000000 || passes<1 || passes>10000 || !std::isfinite(height)||height<=0)
            throw std::invalid_argument("invalid sequential fairing profile");
        std::vector<Point> points;points.reserve(static_cast<std::size_t>(size));
        for (Py_ssize_t i=0;i<size;++i) {
            if (i%32==0) checkpoint(check);
            Ref row(PySequence_GetItem(input,i));Point p;if (!point(row.p,p)) throw std::invalid_argument("finite path required");
            points.push_back(p);
        }
        for (Py_ssize_t pass=0;pass<passes;++pass) for (Py_ssize_t i=1;i<size-1;++i) {
            checkpoint(check);Point candidate;
            for (int axis=0;axis<3;++axis) candidate[axis]=.5*points[i][axis]+.25*(points[i-1][axis]+points[i+1][axis]);
            const auto p=snap(q,candidate,snap_fn);
            if (!p.valid || beyond(difference(p.p,candidate),1e-4,true,compare)) continue;
            Ref row(tuple(p.p));Ref call(PyTuple_Pack(1,row.p));Ref contained(PyObject_CallObject(inside,call.p));
            int valid=PyObject_IsTrue(contained.p);if (valid<0) throw PythonError{};
            if (!valid || std::abs(p.p[1]-candidate[1])>height) continue;
            if (edge(edge_fn,points[i-1],p.p) && edge(edge_fn,p.p,points[i+1])) points[i]=p.p;
        }
        Ref output(PyList_New(0));
        for (const auto& p:points) { Ref row(tuple(p));if (PyList_Append(output.p,row.p)<0) throw PythonError{}; }
        Py_INCREF(output.p);return output.p;
    });
}
} // namespace sim_engine_native
