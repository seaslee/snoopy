/**
 *  \file matrix.h
 *  \brief a template class to describe the Matrix
 *  \author wbd
 *
 *  This class implements the template matrix. It contains the array "data" to
 *  hold the matrix elements.
 *  The matrix can be initialize with the matrix_initializer_list
 *  (a.k, recursive initializer_list).
 *  For example:
 *    Matrix<float, 2> amat {{1,2}, {2, 4}};
 *  The matrix can also initialize with the MatrixShape and the initialize value.
 *  For example:
 *    Matrix<float 2> bmat({2,2}, 0);
 *
 */
#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <fstream>
#include <cblas.h>
#include <cstring>
#include <iomanip>
#include <memory>
#include <initializer_list>
#include <type_traits>
#include <array>
#include "expr.h"
#include "matrix_shape.h"
#include "sub_matrix.h"

//#define DEBUG
namespace matrix {

// ============================================================================
// recursive initializer_list for Matrix Class
template<typename DataType, size_t N>
struct MatrixInit {
  using type = std::initializer_list<typename MatrixInit<DataType, N-1>::type>;
};

template<typename DataType>
struct MatrixInit<DataType, 1> {
  using type = std::initializer_list<DataType>;
};

template<typename DataType>
struct MatrixInit<DataType, 0> ;

template<typename DataType, size_t N>
using matrix_initializer_list = typename MatrixInit<DataType, N>::type;

template<typename DataType, size_t N>
class Matrix : public ExprBase<Matrix<DataType, N>, DataType> {
 public:

  /**
   * default constructor
   */
  Matrix()
      : data(nullptr),
        stride(size_t(0)),
        capicity(size_t(0)),
        row(size_t(0)),
        column(size_t(0)) {
  }
  ;

  /**
   * constructor with matrix shape and initial value
   * @param s is shape of matrix
   * @param n is initial value for the elements in the matrix
   */
  inline Matrix(const MatrixShape<N> &s, const DataType n);

  /**
   * constructor with matrix shape, stride and  initial value
   * @param s is shape of the matrix
   * @param st is stride for each row of the multi-array(Note that: change the shape[N-1] memeory )
   * @param n is initial value for the elements in the matrix
   */
  inline Matrix(const MatrixShape<N> &s, const size_t st, const DataType n);

  /**
   * copy constructor
   * @param m is another matrix to be cloned to this matrix
   */
  inline Matrix(const Matrix<DataType, N> &m);

  /**
   * copy assignment
   * @param m is another matrix to be cloned to this matrix
   * @return the matrix that has copied data from another matrix
   */
  inline Matrix<DataType, N> & operator =(const Matrix<DataType, N> &m);

  /**
   * copy constructor from SubMatrix
   * @param m is the submatrix to be cloned to this matrix
   */
  inline Matrix(const SubMatrix<DataType, N> &m);

  /**
   * copy assignment from SubMatrix
   * @param m is another submatrix to be cloned to this matrix
   * @return the matrix that has copied data from another matrix
   */
  inline Matrix<DataType, N> & operator =(const SubMatrix<DataType, N> &m);

  /**
   * move constructor
   * @param m is another matrix to be moved to this matrix
   */
  inline Matrix(Matrix<DataType, N> &&m);

  /**
   * move assignment operator
   * @param m is matrix to be moved to this matrix
   * @return the matrix that has gotten data from another matrix
   */
  inline Matrix<DataType, N> & operator =(Matrix<DataType, N> &&m);

  /**
   *  constructor with matrix_initializer_list
   * @param t is initializer_list for the matrix
   */
  inline Matrix(matrix_initializer_list<DataType, N> t);

  /**
   *  assignment operator with matrix_initializer_list
   * @param t is initializer_list for the matrix
   * @return the matrix that has filled the data with the initializer_list
   */
  inline Matrix & operator =(matrix_initializer_list<DataType, N> t);
  template<typename T>
  inline Matrix(std::initializer_list<T> t) = delete;
  template<typename T>
  inline Matrix & operator =(std::initializer_list<T> t) = delete;

  /**
   * de-constructor: delete the array
   */
  ~Matrix() {
#ifdef DEBUG
    std::cout<<"decons!!!!!!!!!!!!!!!!!!!!" << this << std::endl;
#endif
    delete[] data;
  }

  /**
   *  index operation for the matrix
   * @param i is the index
   * @return a SubMatrix Object which is refer to some elements in the Matrix Object
   */
  inline SubMatrix<DataType, N - 1> operator[](size_t i) const;
  inline SubMatrix<DataType, N - 1> operator[](size_t i);
  /**
   * slice function for the matrix
   * @param i is the start index
   * @param j is the end index
   * @return a SubMatrix Object which is refer to some elements in the Matrix Object
   */
  inline SubMatrix<DataType, N> slice(size_t i, size_t j) const;
  inline SubMatrix<DataType, N> slice(size_t i, size_t j);

  /**
   * Scalar add Operator
   * @param n is the scalar added to matrix
   * @return the matrix in which the elements are all added the scalar value
   */
  inline Matrix<DataType, N> & operator +=(const DataType & n);

  /**
   * Scalar sub Operator
   * @param n is the scalar subscribed to matrix
   * @return the matrix in which the elements are all subscribed the scalar value
   */
  inline Matrix<DataType, N> & operator -=(const DataType & n);

  /**
   * Scalar multiply Operator
   * @param n is the scalar multiplied to matrix
   * @return the matrix in which the elements are all multiplied the scalar value
   */
  inline Matrix<DataType, N> & operator *=(const DataType & n);

  /**
   * Scalar devision Operator
   * @param n is the scalar divided to matrix
   * @return the matrix in which the elements are all devided the scalar value
   */
  inline Matrix<DataType, N> & operator /=(const DataType & n);

  /**
   * add Operator
   * @param t is the matrix added to this matrix
   * @return the matrix in which the elements: m1(i,j) += m2(i,j)
   */
  inline Matrix<DataType, N> & operator +=(const Matrix<DataType, N> & t);

  /**
   * sub Operator
   * @param t is the matrix subscribed to this matrix
   * @return the matrix in which the elements: m1(i,j) -= m2(i,j)
   */
  inline Matrix<DataType, N> & operator -=(const Matrix<DataType, N> & t);

  /**
   * multiply Operator
   * @param t is the matrix multiplied to this matrix
   * @return the matrix in which the elements: m1(i,j) *= m2(i,j)
   */
  inline Matrix<DataType, N> & operator *=(const Matrix<DataType, N> & t);

  /**
   * division opertor
   * @param t is the matrix divided to this matrix
   * @return the matrix in which the elements: m1(i,j) /= m2(i,j)
   */
  inline Matrix<DataType, N>& operator /=(const Matrix<DataType, N> & t);

  /**
   * Assign Operator
   * @param e is the right matrix object or a scalar
   * @return the result matrix
   */
  template<typename SubType>
  inline Matrix<DataType, N>& operator=(const ExprBase<SubType, DataType> &e);

  /**
   * get the basic element in index (i,j)
   * @param i is the row index
   * @param j is the column index
   * @return the element in the index(i,j)
   */
  inline DataType eval(size_t i, size_t j) const;

  inline void set_row_ele(int i, const Matrix<DataType, N> & s);

  /**
   * get the column size
   * @return the column size
   */
  inline size_t getColumn() const {
    return column;
  }

  /**
   * get the row size
   * @return the row size
   */
  inline size_t getRow() const {
    return row;
  }

  /**
   * get the capicity of the matrix
   * @return the capicity of the matrix
   */
  inline size_t getCapicity() const {
    return capicity;
  }

  /**
   * get the memory length for the matrix
   * @return the capicity of the matrix
   */
  inline int get_capicity() const;
  /**
   * get the number of the elemments in the matrix
   * @return the matrix elements number
   */
  inline int get_size() const;
  /**
   * get the length of the shape
   * @param s is the shape object
   * @param stride is the row length
   * @return the length of the shape with the stride
   */
  inline size_t get_length(const MatrixShape<N> &s, size_t stride) const;
  /**
   * @return the pointer to the data of the matrix
   */
  inline DataType * get_data() {
    return data;
  }
  /**
   * @return the pointer to the data of the matrix
   */
  inline DataType * get_data() const {
    return data;
  }
  //private:
  MatrixShape<N> shape;
  size_t stride;
  size_t capicity;
  DataType * data;
  size_t row;
  size_t column;
};

//=============================================================================
/**
 * One-dimension Matrix Class
 * For example:
 *  Matrix<float, 1> m {1,2,3,4}
 *  support  operator [], such as m[0].
 */
template<typename DataType>
class Matrix<DataType, 1> : public ExprBase<Matrix<DataType, 1>, DataType> {
 public:
  Matrix()
      : data(nullptr),
        stride(size_t(0)) {
  }
  ;
  inline Matrix(const MatrixShape<1> &s, const DataType n);
  inline Matrix(const MatrixShape<1> &s, const size_t st, const DataType n);
  inline Matrix(const Matrix<DataType, 1> &m);
  inline Matrix<DataType, 1> & operator =(const Matrix<DataType, 1> &m);
  inline Matrix(const SubMatrix<DataType, 1> &m);
  inline Matrix<DataType, 1> & operator =(const SubMatrix<DataType, 1> &m);
  inline Matrix(Matrix<DataType, 1> &&m);
  inline Matrix<DataType, 1> & operator =(Matrix<DataType, 1> &&m);
  inline Matrix(matrix_initializer_list<DataType, 1> t);
  inline Matrix & operator =(matrix_initializer_list<DataType, 1> t);
  ~Matrix() {
    delete[] data;
  }
  inline int get_capicity() const;
  inline int get_size() const;
  inline size_t get_length(const MatrixShape<1> &s, size_t stride) const;
  inline DataType * get_data() {
    return data;
  }
  inline DataType * get_data() const {
    return data;
  }
  inline const DataType & operator[](size_t i) const;
  inline DataType & operator[](size_t i);
  inline SubMatrix<DataType, 1> slice(size_t i, size_t j) const;
  inline SubMatrix<DataType, 1> slice(size_t i, size_t j);
  inline Matrix<DataType, 1> & operator +=(const DataType & n);
  inline Matrix<DataType, 1> & operator -=(const DataType & n);
  inline Matrix<DataType, 1> & operator *=(const DataType & n);
  inline Matrix<DataType, 1> & operator /=(const DataType & n);
  inline Matrix<DataType, 1> & operator +=(const Matrix<DataType, 1> & t);
  inline Matrix<DataType, 1> & operator -=(const Matrix<DataType, 1> & t);
  inline Matrix<DataType, 1> & operator *=(const Matrix<DataType, 1> & t);
  inline Matrix<DataType, 1>& operator /=(const Matrix<DataType, 1> & t);
  template<typename SubType>
  inline Matrix<DataType, 1>& operator=(const ExprBase<SubType, DataType> &e);
  inline DataType eval(size_t i, size_t j) const;
  inline void set_row_ele(int i, const Matrix<DataType, 1> & s);
  inline size_t getColumn() const {
    return column;
  }
  inline size_t getRow() const {
    return row;
  }
  inline size_t getCapicity() const {
    return capicity;
  }
  //private:
  MatrixShape<1> shape;
  size_t stride;
  size_t capicity;
  DataType * data;
  size_t row;
  size_t column;
};

//=============================================================================
//matrix operation for two dimensional Matrix and SubMatrix
// matrix multiplication operator

// overload equal Operator
// Ruturn true if all elements in the matrix is equal and stride is equal
template<typename DataType, size_t N>
inline bool operator==(const Matrix<DataType, N>& m1,
                       const Matrix<DataType, N>& m2) {
  DataType * d1 = m1.data;
  DataType * d2 = m2.data;
  if (m1.get_size() != m2.get_size())
    return false;
  for (int i = 0; i < m1.get_size(); ++i) {
    if (d1[i] != d2[i])
      return false;
  }
  return true;
}

template<typename DataType, size_t N>
inline bool operator==(const SubMatrix<DataType, N>& m1,
                       const Matrix<DataType, N>& m2) {
  DataType * d1 = m1.data;
  DataType * d2 = m2.data;
  if (m1.get_size() != m2.get_size())
    return false;
  for (int i = 0; i < m1.get_size(); ++i) {
    if (d1[i] != d2[i])
      return false;
  }
  return true;
}

template<typename DataType, size_t N>
inline bool operator==(const Matrix<DataType, N>& m1,
                       const SubMatrix<DataType, N>& m2) {
  DataType * d1 = m1.data;
  DataType * d2 = m2.data;
  if (m1.get_size() != m2.get_size())
    return false;
  for (int i = 0; i < m1.get_size(); ++i) {
    if (d1[i] != d2[i])
      return false;
  }
  return true;
}

template<typename DataType, size_t N>
inline bool operator==(const SubMatrix<DataType, N>& m1,
                       const SubMatrix<DataType, N>& m2) {
  DataType * d1 = m1.data;
  DataType * d2 = m2.data;
  if (m1.get_size() != m2.get_size())
    return false;
  for (int i = 0; i < m1.get_size(); ++i) {
    if (d1[i] != d2[i])
      return false;
  }
  return true;
}

}  //namespace matrix
#ifdef MATRIX_SCALAR_TYPE_
#error "MATRIX_SCALAR_TYPE_ Should not be defined!"
#endif
#define MATRIX_SCALAR_TYPE_ float
#include "expr-inl.h"
#undef MATRIX_SCALAR_TYPE_
#define MATRIX_SCALAR_TYPE_ double
#include "expr-inl.h"
#undef MATRIX_SCALAR_TYPE_
#define MATRIX_SCALAR_TYPE_ int
#include "expr-inl.h"
#undef MATRIX_SCALAR_TYPE_
#include "matrix-inl.h"
#include "sub_matrix-inl.h"
#include "matrix_math.h"
#include "random.h"
#endif // MATRIX_MATRIX_H

