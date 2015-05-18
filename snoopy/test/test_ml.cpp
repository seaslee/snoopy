/*
 * test_ml.cpp
 *
 *  Created on: 2015年4月27日
 *      Author: wbd
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include "ml/nn.h"
#include "ml/hidden_layer.h"
#include "matrix/utils.h"
#include "ml/activation.h"

using namespace ml;
using namespace std;

int pack(unsigned char zz[4]) {
  return (int) (zz[3]) | (((int) (zz[2])) << 8) | (((int) (zz[1])) << 16)
      | (((int) (zz[0])) << 24);
}

void load(char * datapath, char * labelpath, Matrix<float, 2> & X,
          vector<int> &y, bool is_shuffle = false) {
  //read data
  unsigned int sta;
  FILE * f = fopen(datapath, "rb");
  if (!f) {
    cerr << "Invalid File Name:" << datapath << ", Open Failed" << endl;
    exit(1);
  }
  unsigned char s[4];
  sta = fread(s, 4, 1, f);
  sta = fread(s, 4, 1, f);
  int pic_num = pack(s);
  sta = fread(s, 4, 1, f);
  cout << pic_num << endl;
  int row = pack(s);
  sta = fread(s, 4, 1, f);
  int col = pack(s);
  int d = row * col;
  unsigned char* data = new unsigned char[pic_num * d];
  sta = fread(data, pic_num * d, 1, f);
  fclose(f);
  //read label
  FILE * ff = fopen(labelpath, "rb");
  if (!ff) {
    cerr << "Invalid File Name:" << datapath << ", Open Failed" << endl;
    exit(1);
  }
  sta = fread(s, 4, 1, ff);
  sta = fread(s, 4, 1, f);
  int label_num = pack(s);
  unsigned char* label = new unsigned char[label_num];
  sta = fread(label, label_num, 1, f);
  fclose(ff);
  //put into matrix
  vector<int> ind;
  for (int i = 0; i < pic_num; ++i)
    ind.push_back(i);
  if (is_shuffle)
    random_shuffle(ind.begin(), ind.end());
  for (int i = 0; i < pic_num; ++i) {
    for (int j = 0; j < d; ++j) {
      X[i][j] = static_cast<float>(data[ind[i] * d + j]) / 255.0f;
    }
    y[i] = label[ind[i]];
//        cout << y[i] << endl;
  }
  delete[] data;
  delete[] label;
}

int main() {
  int n = 60000;
  int n1 = 10000;
  Matrix<float, 2> X( { n, 784 }, 0);
  vector<int> y(n, 0);
  Matrix<float, 2> X_test( { n1, 784 }, 0);
  vector<int> y_test(n1, 0);
  load("train-images.idx3-ubyte", "train-labels.idx1-ubyte", X, y, true);
  load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", X_test, y_test,
       false);
  int batch_size = 100;
  float lambda = 0.000001;
  float eta = 0.5;
  NeuralNet<float> nn(batch_size, lambda, eta);
  Layer<float> * l1 = new HiddenLayer<float>(784, 100, batch_size, "relu");
  Layer<float> * l2 = new HiddenLayer<float>(100, 10, batch_size, "softmax");
  //nn.addLayer(l1);
  nn.addLayer(l1);
  nn.addLayer(l2);
  Matrix<float, 2> yy( { n, 10 }, 0);
  oneHotCode(yy, y);
  nn.train(X, yy, 20);
  nn.predict(X, yy);
  cout << "The error on the train data: " << nn.pred_error << endl;
  Matrix<float, 2> yy_test( { n1, 10 }, 0);
  oneHotCode(yy_test, y_test);
  Matrix<float, 2> yy_p( { n1, 10 }, 0);
  nn.predict(X_test, yy_test);
  cout << "The error on the test data: " << nn.pred_error << endl;
  delete l1;
  delete l2;
}

