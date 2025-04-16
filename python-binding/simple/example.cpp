#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.def("add", &add, "addFunction",
          pybind11::arg("i"), pybind11::arg("j"));
}