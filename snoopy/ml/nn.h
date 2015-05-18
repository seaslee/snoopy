#ifndef SNOOPY_MATRIX_NN_H
#define SNOOPY_MATRIX_NN_H

#include <iostream>
#include <vector>
#include "layer.h"
#include "model.h"
#include "matrix/utils.h"

//#define __DEBUG__

using namespace std;
namespace ml {

template<typename DataType>
class NeuralNet : public Model<DataType> {
 public:
  NeuralNet(int b, float l, float et)
      : batch_size(b),
        lambda(l),
        eta(et),
        pred_error(0) {
  }
  ;
  void one_forward(const Matrix<DataType, 2> &input);
  void forward(const Matrix<DataType, 2> & input);
  void backprop(Matrix<DataType, 2> & input, Matrix<DataType, 2> & error);
  void update(float lambda, float eta);
  void train(const Matrix<DataType, 2> & input, const Matrix<DataType, 2> & y,
             int epochs = 100);
  float test(const Matrix<DataType, 2> & input, const Matrix<DataType, 2> & y,
             Matrix<DataType, 2> & y_hat);
  Matrix<DataType, 2> predict(const Matrix<DataType, 2> & input,
                              const Matrix<DataType, 2> & y);
  void addLayer(Layer<DataType> * l) {
    layers.push_back(l);
  }
  ~NeuralNet() {
  }
  ;
  //private:
  vector<Layer<DataType> *> layers;
  int batch_size;  //mini-batch optimize
  float lambda;
  float eta;
  float pred_error;
};

template<typename DataType>
void NeuralNet<DataType>::one_forward(const Matrix<DataType, 2> &input) {
  int layers_size = layers.size();
  if (layers_size == 0) {
    std::cerr << "Neural Net is empty!" << std::endl;
  }
  for (int i = 0; i < layers_size; ++i) {
    if (i == 0) {
      layers[i]->one_forward(input);
    } else {
      layers[i]->one_forward(layers[i - 1]->output1);
    }
  }
}

template<typename DataType>
void NeuralNet<DataType>::forward(const Matrix<DataType, 2> &input) {
  int layers_size = layers.size();
  if (layers_size == 0) {
    std::cerr << "Neural Net is empty!" << std::endl;
  }
  for (int i = 0; i < layers_size; ++i) {
#ifdef __DEBUG__
    cout << "forward in the layer: " << i << endl;
#endif
    if (i == 0) {
      layers[i]->forward(input);
    } else {
      layers[i]->forward(layers[i - 1]->output);
    }
  }
}

template<typename DataType>
void NeuralNet<DataType>::backprop(Matrix<DataType, 2> & input,
                                   Matrix<DataType, 2> & error) {
  int layers_size = layers.size();
  if (layers_size == 0) {
    std::cerr << "Neural Net is empty!" << std::endl;
  }
  for (int i = layers_size - 1; i >= 0; --i) {
#ifdef __DEBUG__
    cout << "!!!!backward in the layer: " << i << endl;
#endif
    if (i == layers_size - 1) {
      layers[i]->backward(layers[i - 1]->output, error, true);
    } else if (i == 0) {
      layers[i]->backward(input, error, false);
    } else {
      layers[i]->backward(layers[i - 1]->output, error, false);
    }
#ifdef __DEBUG__
    cout << "Back Error:::" << endl;
    cout << "Error Before:" << endl;
    cout << error;
#endif
    error = layers[i]->pre_err(error);
#ifdef __DEBUG__
    cout << "Error After:" << endl;
    cout << error;
#endif
    //cout << error ;
  }
}

template<typename DataType>
void NeuralNet<DataType>::update(float lambda, float eta) {
  int layers_size = layers.size();
  if (layers_size == 0) {
    std::cerr << "Neural Net is empty!" << std::endl;
  }
  for (int i = 0; i < layers_size; ++i) {
    layers[i]->update(lambda, eta);
  }
}

template<typename DataType>
void NeuralNet<DataType>::train(const Matrix<DataType, 2> & input,
                                const Matrix<DataType, 2> & y, int epochs) {
  int n = input.row;
  int iter = 0;
  float eta1;
  for (int i = 0; i < epochs; ++i) {
    for (int j = 0; j + batch_size <= n; j += batch_size) {
      Matrix<float, 2> batch_in = input.slice(j, j + batch_size);
      Matrix<float, 2> y_in = y.slice(j, j + batch_size);
      iter = i * n + j + batch_size;
#ifdef __DEBUG__
      cout << "========================================" << endl;
      //forward compute the status for every layer
      cout << "FORWARD::::" << endl;
#endif
      forward(batch_in);
      //compute the error in the output layer
#ifdef __DEBUG__
      cout << "LAST ERROR::::" << endl;
#endif
      Matrix<DataType, 2> error(layers[layers.size() - 1]->output);
      error = layers[layers.size() - 1]->output - y_in;
#ifdef __DEBUG__
      cout << "Last layer output::" << endl;
      cout << layers[layers.size() - 1]->output;
      cout << "true label" << endl;
      cout << y_in;
      cout << "Error:" << endl;
      cout << error;
      //backprop the error to compute the gradient
      cout << "BACKWARD::::" << endl;
#endif
      backprop(batch_in, error);
      //update the weights and bias
#ifdef __DEBUG__
      cout << "UPDATE::::" << endl;
#endif
      update(lambda, eta);
      Matrix<float, 2> y_hat( { y_in.row, y_in.column }, 0);
      float err = test(batch_in, y_in, y_hat);
      cout << "eta: " << eta << " epochs: " << i << ", j: " << j << ", Iter: "
           << iter << ", error: " << err << endl;
    }
  }
}

template<typename DataType>
float NeuralNet<DataType>::test(const Matrix<DataType, 2> & input,
                                const Matrix<DataType, 2> & y,
                                Matrix<DataType, 2> & y_hat) {
  Matrix<DataType, 2> yy(y);
  forward(input);
  y_hat = layers[layers.size() - 1]->output;
  oneHotCode(y_hat);
  vector<int> y_ind(y.row, 0);
  max(y, y_ind, 1);
  float err = 0;
  for (int i = 0; i < y.row; ++i) {
    err += (1.0 - y_hat[i][y_ind[i]]);
  }
  return err;
}

template<typename DataType>
inline Matrix<DataType, 2> NeuralNet<DataType>::predict(
    const Matrix<DataType, 2>& input, const Matrix<DataType, 2> & y) {
  Matrix<DataType, 2> y_p( { y.row, y.column }, 0);
  pred_error = 0;
  for (int i = 0; i < input.row; ++i) {
    Matrix<DataType, 2> x = input.slice(i, i + 1);
    Matrix<DataType, 2> y_in = y.slice(i, i + 1);
    one_forward(x);
    Matrix<DataType, 2> y_hat = layers[layers.size() - 1]->output1;
    oneHotCode(y_hat);
    y_p.set_row_ele(i, y_hat);
    vector<int> y_ind(1, 0);
    max(y_in, y_ind, 1);
    pred_error += (1.0 - y_hat[0][y_ind[0]]);
  }
  return y_p;
}

}

#endif //MATRIX_NN_H
