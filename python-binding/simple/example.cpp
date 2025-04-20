#include <pybind11/pybind11.h>

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example module"; // Optional module docstring

    m.def("add", &add, "A function that adds two numbers",
          pybind11::arg("a"), pybind11::arg("b"));

    m.def("subtract", &subtract, "A function that subtracts two numbers",
          pybind11::arg("a"), pybind11::arg("b"));
}