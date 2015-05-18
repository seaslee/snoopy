/*
 * hidden_layer.h
 *
 *  Created on: 2015年4月27日
 *      Author: wbd
 */

#ifndef SNOOPY_ML_HIDDEN_LAYER_H_
#define SNOOPY_ML_HIDDEN_LAYER_H_

#include "layer.h"
#include "matrix/matrix.h"
#include <iostream>
#include "activation.h"
#include <cmath>

//#define __DEBUG__

namespace ml {

template<typename DataType>
class HiddenLayer : public Layer<DataType> {
 public:
  HiddenLayer()
      : input_num(0),
        output_num(0),
        sample_num(0) {
  }
  ;
  HiddenLayer(int in_n, int out_n, int n, string activation_name)
      : input_num(in_n),
        output_num(out_n),
        sample_num(n) {
    this->input = Matrix<DataType, 2>( { sample_num, output_num }, 0);
    this->output = Matrix<DataType, 2>( { sample_num, output_num }, 0);
    this->output1 = Matrix<DataType, 2>( { 1, output_num }, 0);
    //weights = Matrix<DataType, 2>({input_num, output_num}, 1);
    //weights = Random<DataType, 2>::normal( { input_num, output_num }, 0, 1);
    weights = Random<DataType, 2>::uniform( { input_num, output_num },
                                           -1 / sqrt(in_n), 1 / sqrt(in_n));
    w_grad = Matrix<DataType, 2>(weights);
    bias = Matrix<DataType, 1>( { output_num }, 0);
    b_grad = Matrix<DataType, 1>(bias);
    if (activation_name == "sigmoid") {
      act = new Sigmoid<DataType>;
    } else if (activation_name == "tanh") {
      act = new Tanh<DataType>;
    } else if (activation_name == "softmax") {
      act = new Softmax<DataType>;
    } else if (activation_name == "softsign") {
      act = new Softsign<DataType>;
    } else if (activation_name == "relu") {
      act = new ReLU<DataType>;
    }
  }

  virtual ~HiddenLayer() {
    delete act;
  }
  ;
  void forward(const Matrix<DataType, 2> & input);
  void backward(const Matrix<DataType, 2> & input, Matrix<DataType, 2> & err,
                bool last_layer = false);
  Matrix<DataType, 2> get_output();
  Matrix<DataType, 2> pre_err(const Matrix<DataType, 2> & err);
  void one_forward(const Matrix<DataType, 2> & input);
  void act_fun(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    act->act_fun(t, s);
  }
  void act_fun_de(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    act->act_fun_de(t, s);
  }
  Matrix<DataType, 2> output_err(const Matrix<DataType, 2>& y);
  void update(float lambda, float eta);
  //private:
  Activation<DataType> * act;
  int input_num;
  int output_num;
  int sample_num;
  Matrix<DataType, 2> weights;
  Matrix<DataType, 1> bias;
  Matrix<DataType, 2> w_grad;
  Matrix<DataType, 1> b_grad;
}
;

template<typename DataType>
void HiddenLayer<DataType>::forward(const Matrix<DataType, 2> & input) {
#ifdef __DEBUG__
  cout << "=====================================" <<endl;
  cout << "forward in the hidden layer" << endl;
  cout << "Input: " <<endl;
  cout << input;
#endif
  this->input = dot(input, weights);
  Matrix<DataType, 2> rep_bias = repmat(bias, input.row);
  this->input = this->input + rep_bias;
  act_fun(this->output, this->input);
  //act_fun(this->input, this->input);  //TODO
  //copyMatrix(this->output, this->input);
#ifdef __DEBUG__
  cout << "weights: " <<endl;
  cout << weights;
  cout << "bias:" <<endl;
  cout << bias;
  cout << "Input= input * weights + bias" <<endl;
  cout << this->input;
  cout << "Output= tanh(input * weights + bias)" <<endl;
  cout << this->output;
#endif
}

template<typename DataType>
void HiddenLayer<DataType>::one_forward(const Matrix<DataType, 2> & input) {
  this->input = dot(input, weights);
  Matrix<DataType, 2> rep_bias = repmat(bias, input.row);
  this->input = this->input + rep_bias;
  act_fun(this->output1, this->input);
}

template<typename DataType>
void HiddenLayer<DataType>::backward(
    const Matrix<DataType, 2> & lastlayer_output, Matrix<DataType, 2> & pre_err,
    bool lastest_layer) {
#ifdef __DEBUG__
  cout << "=====================================" <<endl;
  cout << "backward in the hidden layer" << endl;
#endif
  if (!lastest_layer) {
    Matrix<DataType, 2> input_derivate(this->output);  //TODO
    act_fun_de(input_derivate, this->output);
#ifdef __DEBUG__
    cout << "last layer error ::" <<endl;
    cout << err;
#endif
    pre_err = input_derivate * pre_err;
#ifdef __DEBUG__
    cout << "Input in this layer::" <<endl;
    cout << this->input;
    cout << "Derivate of Input::" <<endl;
    cout << input_derivate;
    cout << "error = derivate * last_error" <<endl;
    cout << err;
#endif
  }
  auto out_t = transpose(lastlayer_output);
#ifdef __DEBUG__
  cout << "Last Layer output::" <<endl;
  cout << input;
  cout << "Error in this layer::" <<endl;
  cout << err;
#endif
  w_grad = dot(out_t, pre_err);
  b_grad = sum(pre_err);
#ifdef __DEBUG__
  cout << "W_grad = last_layer' * error" <<endl;
  cout << w_grad;
  cout << "Bias= sum(error)"<<endl;
  cout << b_grad;
#endif
}

template<typename DataType>
Matrix<DataType, 2> HiddenLayer<DataType>::pre_err(
    const Matrix<DataType, 2> & err) {
  auto w_t = transpose(weights);
#ifdef __DEBUG__
  cout << "Back Error, Weights" <<endl;
  cout << w_t;
#endif
  return dot(err, w_t);
}

template<typename DataType>
void HiddenLayer<DataType>::update(float lambda, float eta) {
#ifdef __DEBUG__
  cout << "Before the wegiths and bias::" << eta << endl;
  cout << "weights::"<<endl;
  cout << weights;
  cout << "bias" <<endl;
  cout << bias;
  cout << "W_grad" <<endl;
  cout << w_grad;
  cout << "Bias_grad" <<endl;
  cout << b_grad;
#endif
  weights = weights - eta * (w_grad / sample_num + lambda * weights);
  bias = bias - eta * (b_grad / sample_num);  //NOTE BUG, Matrix libarary
#ifdef __DEBUG__
      cout << "After the wegiths and bias::" << eta << endl;
      cout << "weights::"<<endl;
      cout << weights;
      cout << "bias" <<endl;
      cout << bias;
#endif
}

}
#endif /* ML_HIDDEN_LAYER_H_ */
