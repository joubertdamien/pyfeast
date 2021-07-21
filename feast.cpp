#include "feast.hpp"

std::unique_ptr<FeastLayer> filt_;

static PyObject *init(PyObject *self, PyObject *args){
    uint16_t x, y, w, nb;
    uint64_t tau;
    float_t th_open, th_close, lr;
    if (!PyArg_ParseTuple(args, "HHHkfffH", &x, &y, &w, &tau, &th_open, &th_close, &lr, &nb))
        return NULL;
    filt_ =  std::make_unique<FeastLayer>(x, y, w, tau, th_open, th_close, lr, nb);
    return Py_True;
}

static PyObject *filter(PyObject *self, PyObject *args){
    PyObject *chunk, *ex_chunk; 
    if (!PyArg_ParseTuple(args, "OO", &chunk, &ex_chunk))
        return NULL;
    PyArrayObject *chunk_array, *ex_chunk_array; 
    chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(chunk, NPY_ARRAY_C_CONTIGUOUS));
    ex_chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(ex_chunk, NPY_ARRAY_C_CONTIGUOUS));
    auto shape = PyArray_SHAPE(chunk_array);
    auto type = PyArray_DTYPE(ex_chunk_array);
    std::vector<Event> in(shape[0]);
    std::vector<NeuronEvent> out;
    std::memcpy(in.data(), PyArray_DATA(chunk_array), sizeof(Event) * shape[0]);
    filt_->process(in, out);
    npy_intp new_size = static_cast<npy_intp>(out.size());
    if(out.size() == 0)
        return Py_False;
    auto result_array = PyArray_NewFromDescr(&PyArray_Type, type, 1, &new_size, nullptr, nullptr, 0, nullptr);
    std::memcpy(PyArray_DATA((PyArrayObject*) result_array ), out.data(), sizeof(NeuronEvent) * new_size);
    auto stream = PyDict_New();
    PyDict_SetItem(stream, PyUnicode_FromString("ev"), result_array);
    return stream;
}

static PyObject *getNeuronState(PyObject *self, PyObject *args){
    std::vector<float_t> wei, ths;
    filt_->getThs(ths);
    filt_->getWeights(wei);
    npy_intp new_size_wei = static_cast<npy_intp>(wei.size());
    npy_intp new_size_ths = static_cast<npy_intp>(ths.size());
    PyObject *ths_array = PyArray_SimpleNew(1, &new_size_ths, NPY_FLOAT32);
    PyObject *wei_array = PyArray_SimpleNew(1, &new_size_wei, NPY_FLOAT32);
    std::memcpy(PyArray_DATA((PyArrayObject*) ths_array), ths.data(), sizeof(float_t) * new_size_ths);
    std::memcpy(PyArray_DATA((PyArrayObject*) wei_array), wei.data(), sizeof(float_t) * new_size_wei);
    auto stream = PyDict_New();
    PyDict_SetItem(stream, PyUnicode_FromString("weights"), wei_array);
    PyDict_SetItem(stream, PyUnicode_FromString("ths"), ths_array);
    Py_DecRef(ths_array);
    Py_DecRef(wei_array);
    return stream;
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
    "tau: uint64_t\n"
    "   Time decay of the time surface to compute the local context\n"
    "th_open: float_t\n"
    "   threshold update when neither neurons reacts\n"
    "th_close: float_t\n"
    "   threshold update when a neuron spikes\n"
    "lr: float_t\n"
    "   learning rate\n"
    "uint16_t: float_t\n"
    "   Number of neurons in the layer\n"
    );

PyDoc_STRVAR(
    filter_doc,
    "Filter the input buffer of event and returns the ouput events.\n\n"
    "Parameters\n"
    "----------\n"
    "in: Numpy structured array of Event\n"
    "   Input buffer\n"
    "out: Numpy structured array of NeuronEvent\n"
    "   Empty buffer. Mainly used to catch the type\n"
    "Returns\n"
    "----------\n"
    "res: Python dictionnary\n"
    "   res[""ev""] contains a structed array of NeuronEvent\n"
    );
PyDoc_STRVAR(
    getNeuronState_doc,
    "Return the neurons weights and thresholds in a Python dictionnary.\n\n"
    "Returns\n"
    "----------\n"
    "res: Python dictionnary\n"
    "   res[""weights""] contains the weights\n"
    "   res[""th""] contains the thresholds\n"
    );
static char module_docstring[] = "C++ Feast implementation";

//Module specification
static PyMethodDef module_methods[] = {
    {"filter", (PyCFunction)filter, METH_VARARGS, init_doc},
    {"getNeuronState", (PyCFunction)getNeuronState, METH_VARARGS, getNeuronState_doc},
    {"init", (PyCFunction)init, METH_VARARGS, filter_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef def = {
    PyModuleDef_HEAD_INIT,
    "pyfeast",
    module_docstring,
    -1,
    module_methods};

//Initialize module
PyMODINIT_FUNC PyInit_pyfeast(void){
    PyObject *m = PyModule_Create(&def);
    if (m == NULL)
        return NULL;
    //numpy functionallity
    import_array();
    return m;
}
