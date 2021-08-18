#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <cstring>
#include <numpy/arrayobject.h>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <limits>
#include <memory>
#include <cmath>
#include "sepia/source/sepia.hpp"
#include "feast.hpp"
#ifndef IMPFEAST_HPP_
#define IMPFEAST_HPP_
float_t closeCos(float_t& th, float_t& th_close){
    return th + th_close * std::cos(th * M_PI_2);
}
float_t close1(float_t& th, float_t& th_close){
    return th + th_close * (1-th);
}
float_t close2(float_t& th, float_t& th_close){
    return th + th_close * std::pow(1-th, 2);
}
#endif 