#ifndef SNOOPY_MATRIX_MODEL_H
#define SNOOPY_MATRIX_MODEL_H

#include "matrix/matrix.h"

using namespace matrix;

namespace ml {

template<typename DataType>
class Model {
 public:
  virtual void train(const Matrix<DataType, 2> & input,
      const Matrix<DataType, 2> & y, int epochs =
      100)=0;
  virtual Matrix<DataType, 2> predict(const Matrix<DataType, 2> & input, const Matrix<DataType, 2> & y)=0;
  virtual ~Model() {
  }
};
}
#endif //MATRIX_MODEL_H
