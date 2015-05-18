#ifndef MATRIX_RANDOM_H
#define MATRIX_RANDOM_H

#include <random>
#include "matrix.h"
using namespace matrix;
namespace matrix {

//generate matrix with random values
template<typename DataType, size_t N>
class Random {
 public:

  //uniform sampling
  static Matrix<DataType, N> uniform(const MatrixShape<N> s, float a = 0,
                                     float b = 1) {
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_real_distribution<> dis(a, b);
    Matrix<DataType, N> mat(s, 0);
    DataType * data = mat.get_data();
    for (size_t i = 0; i < mat.getCapicity(); ++i) {
      data[i] = dis(gen);
    }
    return mat;
  }

  //guassian sampling
  static Matrix<DataType, N> normal(const MatrixShape<N> s, float a = 0,
                                    float b = 1) {
    std::random_device r;
    std::mt19937 gen(r());
    std::normal_distribution<> dis(a, b);
    Matrix<DataType, N> mat(s, 0);
    DataType * data = mat.get_data();
    for (size_t i = 0; i < mat.getCapicity(); ++i) {
      data[i] = dis(gen);
    }
    return mat;
  }

};

}
#endif //MATRIX_RANDOM_H
