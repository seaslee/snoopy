//============================================================================
// Name        : Matrix.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include "matrix/matrix.h"

using namespace matrix;
using namespace std;

struct Square {
  static double matrix_op(double x) {
    return (x * x);
  }
};

void test_ele_op() {
  Matrix<float, 3> m1 { { { 1, 2, 3 }, { 2, 3, 4 } },
      { { 1, 2, 3 }, { 2, 3, 4 } } };
  Matrix<float, 3> m2 { { { 1, 4, 3 }, { 6, 3, 4 } },
      { { 1, 4, 3 }, { 6, 3, 4 } } };
  check(!(m1 == m2), "m1 m2 is  equal!");
  Matrix<float, 3> m3(m1);
  m3 = m1 + m2;
  Matrix<float, 3> m4 = { { { 2, 6, 6 }, { 8, 6, 8 } }, { { 2, 6, 6 },
      { 8, 6, 8 } } };
  check(m3 == m4, "add: m3 m4 is not equal!");
  m3 = m1 - m2;
  m4 = { { {0, -2, 0}, {-4,0,0}}, { {0,-2,0}, {-4,0,0}}};
  check(m3 == m4, "sub: m3 m4 is not equal!");
  m3 = m1 * m2;
  m4 = { { {1, 8, 9}, {12,9,16}}, { {1,8,9}, {12,9,16}}};
  check(m3 == m4, "mul: m3 m4 is not equal!");
  m4 = m4 - 3;
  Matrix<float, 3> m5 = { { {-2, 5, 6}, {9,6,13}}, { {-2,5,6}, {9,6,13}}};
  check(m4 == m5, "mul: m3 m4 is not equal!");
}

void test_dot() {
  Matrix<float, 2> m1 { { 1, 2, 3 }, { 2, 3, 4 } };
  Matrix<float, 2> m2 { { 2, 3, 4 }, { 2, 3, 4 } };
  Matrix<float, 2> m3 = dot(m1, m2, 1.f, false);
  Matrix<float, 2> m4 = dot(m1, m2);
  cout << m3;
  cout << m4;
  check(m3 == m4, "mul: m3 m4 is not equal!");
  Matrix<float, 2> m5 = transpose(m1);
  //Matrix<float, 2> m6 = transpose(m1, 1.f, true);
  Matrix<float, 2> m6 { { 1, 2 }, { 2, 3 }, { 3, 4 } };
  check(m5 == m6, "mul: m3 m4 is not equal!");
  Matrix<float, 3> mm1 { { { 1, 2, 3 }, { 2, 3, 4 } }, { { 1, 2, 3 },
      { 2, 3, 4 } } };
  SubMatrix<float, 2> sm1 = mm1[0];
  Matrix<float, 2> m7 = dot(sm1, m2);
  check(m4 == m7, "mul: m3 m4 is not equal!");
}

void test_math() {
  Matrix<float, 2> m1 { { 1, 2, 3 }, { 2, 3, 4 } };
  Matrix<float, 1> m2 = sum(m1, 0);
  Matrix<float, 1> temp { 3, 5, 7 };
  check(m2 == temp, "not equal!");
  Matrix<float, 1> m3 = sum(m1, 1);
  Matrix<float, 1> temp1 { 6, 9 };
  check(m3 == temp1, "not equal!");
  Matrix<float, 1> m4 = sum(m1, 2);
  Matrix<float, 1> temp2 { 15 };
  check(m4 == temp2, "not equal!");
//  Matrix<float, 1> temp {1,2,3};
//  cout << temp[0] << "!!!" << temp[1] <<endl;
}

int main(void) {
  test_ele_op();
  //test_dot();
  //test_math();
  Matrix<float, 2> s( { 1, 2 }, 0);
  cout << "Test Sucess" << endl;
  //cout << s[0][0] << " !!! " << s[0][1] <<endl;
  return EXIT_SUCCESS;
}
