#include "filter.hpp"

std::unique_ptr<Filter> filt_;

static PyObject *init(PyObject *self, PyObject *args){
    uint16_t x, y, w, th_ev;
    uint64_t th_time;
    if (!PyArg_ParseTuple(args, "HHHkH", &x, &y, &w, &th_time, &th_ev))
        return NULL;
    filt_ =  std::make_unique<Filter>(x, y, w, th_time, th_ev);
    return Py_True;
}

static PyObject *filter(PyObject *self, PyObject *args){
    PyObject *chunk; 
    if (!PyArg_ParseTuple(args, "O", &chunk))
        return NULL;
    PyArrayObject *chunk_array; 
    chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(chunk, NPY_ARRAY_C_CONTIGUOUS));
    auto shape = PyArray_SHAPE(chunk_array);
    auto type = PyArray_DTYPE(chunk_array);
    std::vector<Event> in(shape[0]), out;
    std::memcpy(in.data(), PyArray_DATA(chunk_array), sizeof(Event) * shape[0]);
    filt_->process(in, out);
    npy_intp new_size = static_cast<npy_intp>(out.size());
    if(out.size() == 0)
        return Py_False;
    PyObject *result_array = PyArray_NewFromDescr(&PyArray_Type, type, 1, &new_size, nullptr, nullptr, 0, nullptr);
    std::memcpy(PyArray_DATA((PyArrayObject*) result_array ), out.data(), sizeof(Event) * new_size);
    return (PyObject*)result_array;
}

// Doc
PyDoc_STRVAR(
    init_doc,
    "Initialize the filter.\n\n"
    "Parameters\n"
    "----------\n"
    "x: uint16_t\n"
    "   Number of rows\n"
    "y: uint16_t\n"
    "   Number of columns\n"
    "w: uint16_t\n"
    "   Half size of the square ROI\n"
    "   Final size: (2*w+1)**2\n"
    "time_th: uint16_t\n"
    "   Size of the temporal filtering\n"
    "ev_th: numpy_array\n"
    "   Number of events needed in the ROI*time_th volume to pass the filter\n");
PyDoc_STRVAR(
    filter_doc,
    "Filter the input buffer of event and returns the ouput events.\n\n"
    "Parameters\n"
    "----------\n"
    "in: std::vector<Event>\n"
    "   Input buffer\n"
    "out: std::vector<Event>\n"
    "   Output buffer\n");
static char module_docstring[] = "C++ Template filter";

//Module specification
static PyMethodDef module_methods[] = {
    {"filter", (PyCFunction)filter, METH_VARARGS, init_doc},
    {"init", (PyCFunction)init, METH_VARARGS, filter_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef def = {
    PyModuleDef_HEAD_INIT,
    "filter",
    module_docstring,
    -1,
    module_methods};

//Initialize module
PyMODINIT_FUNC PyInit_filter(void){
    PyObject *m = PyModule_Create(&def);
    if (m == NULL)
        return NULL;
    //numpy functionallity
    import_array();
    return m;
}
