#include "feast.hpp"
#include "impfeast.hpp"
std::unique_ptr<FeastLayer<FeastNeuron<closeId>>> filt_;
std::unique_ptr<FeastLayer<FeastNeuron<closeCos>>> filt_cos_;
std::unique_ptr<FeastLayer<FeastNeuron<close1>>> filt_1_;
std::unique_ptr<FeastLayer<FeastNeuron<close2>>> filt_2_;
static PyObject *init(PyObject *self, PyObject *args){
    uint16_t x, y, w, nb;
    uint64_t tau;
    float_t th_open, th_close, lr;
    char* s;
    if (!PyArg_ParseTuple(args, "HHHKfffHs", &x, &y, &w, &tau, &th_open, &th_close, &lr, &nb, &s))
        return NULL;
    switch(s[0]){
        case 'f':
            filt_ =  std::make_unique<FeastLayer<FeastNeuron<closeId>>>(x, y, w, tau, th_open, th_close, lr, nb);
        break;
        case 'c':
            filt_cos_ =  std::make_unique<FeastLayer<FeastNeuron<closeCos>>>(x, y, w, tau, th_open, th_close, lr, nb);
        break;
        case '1':
            filt_1_ =  std::make_unique<FeastLayer<FeastNeuron<close1>>>(x, y, w, tau, th_open, th_close, lr, nb);
        break;
        case '2':
            filt_2_ =  std::make_unique<FeastLayer<FeastNeuron<close2>>>(x, y, w, tau, th_open, th_close, lr, nb);
        break;
        default:
        break;
    }
    return Py_True;
}
static PyObject *filter(PyObject *self, PyObject *args){
    PyObject *chunk, *ex_chunk; 
    char * s;
    if (!PyArg_ParseTuple(args, "OOs", &chunk, &ex_chunk, &s))
        return NULL;
    PyArrayObject *chunk_array, *ex_chunk_array; 
    chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(chunk, NPY_ARRAY_C_CONTIGUOUS));
    ex_chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(ex_chunk, NPY_ARRAY_C_CONTIGUOUS));
    auto shape = PyArray_SHAPE(chunk_array);
    auto type = PyArray_DTYPE(ex_chunk_array);
    std::vector<Event> in(shape[0]);
    std::vector<NeuronEvent> out;
    std::memcpy(in.data(), PyArray_DATA(chunk_array), sizeof(Event) * shape[0]);
    switch(s[0]){
        case 'f':
            filt_->process(in, out);
        break;
        case 'c':
            filt_cos_->process(in, out);
        break;
        case '1':
            filt_1_->process(in, out);
        break;
        case '2':
            filt_2_->process(in, out);
        break;
        default:
        break;
    }
    npy_intp new_size = static_cast<npy_intp>(out.size());
    if(out.size() == 0)
        return Py_False;
    auto result_array = PyArray_NewFromDescr(&PyArray_Type, type, 1, &new_size, nullptr, nullptr, 0, nullptr);
    std::memcpy(PyArray_DATA((PyArrayObject*) result_array ), out.data(), sizeof(NeuronEvent) * new_size);
    auto stream = PyDict_New();
    PyDict_SetItem(stream, PyUnicode_FromString("ev"), result_array);
    return stream;
}

static PyObject *filterIndex(PyObject *self, PyObject *args){
    PyObject *chunk, *ex_chunk; 
    char * s;
    if (!PyArg_ParseTuple(args, "OOs", &chunk, &ex_chunk, &s))
        return NULL;
    PyArrayObject *chunk_array, *ex_chunk_array; 
    chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(chunk, NPY_ARRAY_C_CONTIGUOUS));
    ex_chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(ex_chunk, NPY_ARRAY_C_CONTIGUOUS));
    auto shape = PyArray_SHAPE(chunk_array);
    auto type = PyArray_DTYPE(ex_chunk_array);
    std::vector<Event> in(shape[0]);
    std::vector<NeuronEvent> out;
    std::memcpy(in.data(), PyArray_DATA(chunk_array), sizeof(Event) * shape[0]);
    switch(s[0]){
        case 'f':
            filt_->processIndex(in, out);
        break;
        case 'c':
            filt_cos_->processIndex(in, out);
        break;
        case '1':
            filt_1_->processIndex(in, out);
        break;
        case '2':
            filt_2_->processIndex(in, out);
        break;
        default:
        break;
    }
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
        char * s;
    if (!PyArg_ParseTuple(args, "s", &s))
        return NULL;
    std::vector<float_t> wei, ths, ent;
    switch(s[0]){
        case 'f':
            filt_->getThs(ths);
            filt_->getWeights(wei);
            filt_->getEnt(ent);
        break;
        case 'c':
            filt_cos_->getThs(ths);
            filt_cos_->getWeights(wei);
            filt_cos_->getEnt(ent);
        break;
        case '1':
            filt_1_->getThs(ths);
            filt_1_->getWeights(wei);
            filt_1_->getEnt(ent);
        break;
        case '2':
            filt_2_->getThs(ths);
            filt_2_->getWeights(wei);
            filt_2_->getEnt(ent);
        break;
        default:
        break;
    }
    npy_intp new_size_wei = static_cast<npy_intp>(wei.size());
    npy_intp new_size_ths = static_cast<npy_intp>(ths.size());
    npy_intp new_size_ent = static_cast<npy_intp>(ent.size());
    PyObject *ths_array = PyArray_SimpleNew(1, &new_size_ths, NPY_FLOAT32);
    PyObject *wei_array = PyArray_SimpleNew(1, &new_size_wei, NPY_FLOAT32);
    PyObject *ent_array = PyArray_SimpleNew(1, &new_size_ent, NPY_FLOAT32);
    std::memcpy(PyArray_DATA((PyArrayObject*) ths_array), ths.data(), sizeof(float_t) * new_size_ths);
    std::memcpy(PyArray_DATA((PyArrayObject*) wei_array), wei.data(), sizeof(float_t) * new_size_wei);
    std::memcpy(PyArray_DATA((PyArrayObject*) ent_array), ent.data(), sizeof(float_t) * new_size_ent);
    auto stream = PyDict_New();
    PyDict_SetItem(stream, PyUnicode_FromString("weights"), wei_array);
    PyDict_SetItem(stream, PyUnicode_FromString("ths"), ths_array);
    PyDict_SetItem(stream, PyUnicode_FromString("ent"), ent_array);
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
    "nb: uint16_t\n"
    "   Number of neurons in the layer\n"
    "s: char\n"
    "   Type of neuron addressed\n"
    );
PyDoc_STRVAR(
    filter_doc,
    "Filter the input buffer of event using a time surface and returns the ouput events.\n\n"
    "Parameters\n"
    "----------\n"
    "in: Numpy structured array of Event\n"
    "   Input buffer\n"
    "out: Numpy structured array of NeuronEvent\n"
    "   Empty buffer. Mainly used to catch the type\n"
    "s: char\n"
    "   Type of neuron addressed\n"
    "Returns\n"
    "----------\n"
    "res: Python dictionnary\n"
    "   res[""ev""] contains a structed array of NeuronEvent\n"
    );
PyDoc_STRVAR(
    filter_index_doc,
    "Filter the input buffer of event using en index surface and returns the ouput events.\n\n"
    "Parameters\n"
    "----------\n"
    "in: Numpy structured array of Event\n"
    "   Input buffer\n"
    "out: Numpy structured array of NeuronEvent\n"
    "   Empty buffer. Mainly used to catch the type\n"
    "s: char\n"
    "   Type of neuron addressed\n"
    "Returns\n"
    "----------\n"
    "res: Python dictionnary\n"
    "   res[""ev""] contains a structed array of NeuronEvent\n"
    );
PyDoc_STRVAR(
    getNeuronState_doc,
    "Return the neurons weights and thresholds in a Python dictionnary.\n\n"
    "Parameters\n"
    "----------\n"
    "s: char\n"
    "   Type of neuron addressed\n"
    "Returns\n"
    "----------\n"
    "res: Python dictionnary\n"
    "   res[""weights""] contains the weights\n"
    "   res[""th""] contains the thresholds\n"
    "   res[""ent""] contains the entropy of the weight\n"
    );
static char module_docstring[] = "C++ Feast implementation";

//Module specification
static PyMethodDef module_methods[] = {
    {"filter", (PyCFunction)filter, METH_VARARGS, filter_doc},
    {"filterIndex", (PyCFunction)filterIndex, METH_VARARGS, filter_index_doc},
    {"getNeuronState", (PyCFunction)getNeuronState, METH_VARARGS, getNeuronState_doc},
    {"init", (PyCFunction)init, METH_VARARGS, init_doc},
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
