#ifndef MATRIX_UTILS_H_
#define MATRIX_UTILS_H_
#include <iostream>
#include <string>
#include <assert.h>

using namespace std;
// check condition is satisified
// assert with condition and output the error string
void check(int condition, string s) {
  if (!condition) {
    cerr << s << endl;
    exit(-1);
  }
}

void check_shape(int condition, string s) {
  if (!condition) {
    cerr << s << endl;
    exit(-1);
  }
}

#endif //MATRIX_UTILS_H_
