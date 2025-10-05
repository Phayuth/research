#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

std::vector<std::vector<double>> plan(const std::vector<double> &qa,
                                      const std::vector<double> &qb) {
    std::vector<std::vector<double>> path;
    for (size_t i = 0; i < 10; i++) {
        std::vector<double> point(6);
        for (size_t j = 0; j < 6; j++) {
            point[j] = qa[j] + qb[j];
        }
        path.push_back(point);
    }
    return path;
}

PYBIND11_MODULE(my_module, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function that adds two numbers");
    m.def("plan", &plan, "A function that plans a path between 2 configs");
}