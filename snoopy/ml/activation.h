/*
 * activation.h
 *
 *  Created on: May 11, 2015
 *      Author: wbd
 */

#ifndef SNOOPY_ML_ACTIVATION_H_
#define SNOOPY_ML_ACTIVATION_H_

#include "matrix/matrix.h"
#include <cfloat>
#include <iostream>

namespace ml {

template<typename DataType>
class Activation {
 public:
  virtual void act_fun(Matrix<DataType, 2> & t,
                       const Matrix<DataType, 2> & s)=0;
  virtual void act_fun_de(Matrix<DataType, 2> & t,
                          const Matrix<DataType, 2> & s)=0;
  virtual ~Activation() {
  }
  ;
};

template<typename DataType>
struct sigmoid {
  inline static DataType matrix_op(DataType x) {
    return 1.0f / (1 + exp(-x));
  }
};

template<typename DataType>
class Sigmoid : public Activation<DataType> {
 public:
  Sigmoid() {
  }
  ;
  virtual ~Sigmoid() {
  }
  ;
  void act_fun(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    t = single_op<sigmoid<DataType>>(s);
  }
  void act_fun_de(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    //t = single_op<sigmoid>(s) * (1 - single_op<sigmoid>(s));
    t = s * (1 - s);
  }
};

template<typename DataType>
struct tanh_map {
  inline static DataType matrix_op(DataType x) {
    return tanh(x);
  }
};

template<typename DataType>
class Tanh : public Activation<DataType> {
 public:
  Tanh() {
  }
  ;
  virtual ~Tanh() {
  }
  ;
  const float a = 1;  //1.7159;
  const float b = 1;  //0.6667;
  void act_fun(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    t = single_op<tanh_map<DataType>>(s);  //TODO
  }

  void act_fun_de(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    //t = a * b * (1 - single_op<tanh_map>(b * s) * single_op<tanh_map>(b * s));
    //t = (1 - s) * (1 + s); //TODO
    t = 1 - s * s;
  }
};

template<typename DataType>
struct softsign {
  inline static DataType matrix_op(DataType x) {
#ifdef USE_DOUBLE
    if (x < DBL_MAX)
    return x > -DBL_MAX ? x / (1 + abs(x)) : -1;
    return 1;
#else
    if (x < FLT_MAX)
      return x > -FLT_MAX ? x / (1 + abs(x)) : -1;
    return 1;
#endif
  }
};

template<typename DataType>
struct fabs_map {
  inline static DataType matrix_op(DataType x) {
    return fabs(x);
  }
};

template<typename DataType>
struct square_map {
  inline static DataType matrix_op(DataType x) {
    return x * x;
  }
};

template<typename DataType>
class Softsign : public Activation<DataType> {
 public:
  Softsign() {
  }
  ;
  virtual ~Softsign() {
  }
  ;
  void act_fun(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    t = single_op<softsign<DataType>>(s);
  }

  void act_fun_de(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    t = single_op<square_map<DataType>>(1 - single_op<fabs_map<DataType>>(s));
  }
};

template<typename DataType>
struct rectfier {
  inline static DataType matrix_op(DataType x) {
    return std::max(static_cast<DataType>(0), x);
  }
};

template<typename DataType>
struct rectfier_deri_map {
  inline static DataType matrix_op(DataType x) {
    return x > 0 ? 1 : 0;
  }
};

template<typename DataType>
class ReLU : public Activation<DataType> {
 public:
  ReLU() {
  }
  ;
  virtual ~ReLU() {
  }
  ;
  void act_fun(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    t = single_op<rectfier<DataType>>(s);
  }

  void act_fun_de(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    t = single_op<rectfier_deri_map<DataType>>(s);
  }
};

//Softmax is the activation function for the output layer
template<typename DataType>
class Softmax : public Activation<DataType> {
 public:
  Softmax() {
  }
  ;
  virtual ~Softmax() {
  }
  ;
  void act_fun(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    t = softmax(s);
  }
  void act_fun_de(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {

  }
};

}

#endif /* ML_ACTIVATION_H_ */
