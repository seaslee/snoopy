#ifndef SNOOPY_ML_LAYER_H_
#define SNOOPY_ML_LAYER_H_

#include "matrix/matrix.h"

using namespace matrix;

namespace ml {
template<typename DataType>
class Layer {
 public:
  virtual void one_forward(const Matrix<DataType, 2> & input) =0;
  virtual void forward(const Matrix<DataType, 2> & input)=0;
  virtual void backward(const Matrix<DataType, 2> & input,
                        Matrix<DataType, 2> & err, bool last_layer = false)=0;
  virtual Matrix<DataType, 2> pre_err(const Matrix<DataType, 2> & err)=0;
  virtual void update(float lambda, float eta)=0;
  virtual ~Layer() {
  }
  Matrix<DataType, 2> output;
  Matrix<DataType, 2> input;
  Matrix<DataType, 2> output1;
};
}

#endif /*ML_LAYER_H_*/
