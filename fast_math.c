/**********************************************************************************************
* METHOD A: Python C-API logic, This function handles Python Objects directly.               *
* Python and C live in two different worlds. Python is a "High-Level" language               *
* (everything is a complex object with metadata), while C is a "Low-Level" language          *
* (everything is raw bytes and memory addresses). This code is the "Translation Layer"       *
* (Boilerplate) required for them to talk to each other.                                     *
* COMPILE: gcc -shared -o fast_math.so -fPIC $(python3-config --includes) fast_math.c        *
**********************************************************************************************/
#define PY_SSIZE_T_CLEAN                                           // use Py_ssize_t (64-bit integers) for all lengths
#include <Python.h>

// 1. The Calculation Function
static PyObject* method_fast_sum(PyObject* self, PyObject* args) { // PyObject* self: Points to the module itself
    PyObject* list_obj;                                            // PyObject* args: This is a Python Tuple where arguments are all packed 
    if (!PyArg_ParseTuple(args, "O", &list_obj)) {                 // PyArg_ParseTuple: unpack arguments
        return NULL;                                               
    }
    if (!PyList_Check(list_obj)) {                                 // PyList_Check: this piece of memory is a Python List
        PyErr_SetString(PyExc_TypeError, "Parameter must be a list.");
        return NULL;
    }
    long sum = 0;
    Py_ssize_t n = PyList_Size(list_obj);                          // PyList_Size: get the count
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PyList_GetItem(list_obj, i);              // PyList_GetItem: the item at index i (This is still a PyObject*)
        sum += PyLong_AsLong(item);                                // PyLong_AsLong: converts the object into a raw 64-bit number 
    }                                                              //                that C can actually add to a variable.
    return PyLong_FromLong(sum);                                   // PyLong_FromLong: converts the number into a Python Object container
}

// 2. The Method Table. This table maps the Python name to the C function address.
static PyMethodDef FastMathMethods[] = {                           // Python_Name, C_fce_Pointer, Type_of_Args, Docstring
    {"fast_sum", method_fast_sum, METH_VARARGS, "Calculate sum of a list quickly."},  // METH_VARARGS - tuple of arguments
    {NULL, NULL, 0, NULL}                                          // The "Sentinel" (VERY IMPORTANT) C arrays don't know their own length.
};

// 3. The Module Definition
static struct PyModuleDef fastmathmodule = {
    PyModuleDef_HEAD_INIT,                        // Standard internal header
    "fast_math",                                  // The name used in: import fast_math
    "C optimized math functions for Python",      // The module's description
    -1,                                           // Global state flag (-1 = simple/stateless)
    FastMathMethods                               // Points back to the Method Table
};

// 4. The Entry Point (PyInit_fast_math) This is what Python looks for during 'import fast_math'
PyMODINIT_FUNC PyInit_fast_math(void) {           // The Naming Rule - it is fixed
    return PyModule_Create(&fastmathmodule);      // Entry Point calls PyModule_Create()
}
