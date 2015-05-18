#ifndef MATRIX_INL_H_
#define MATRIX_INL_H_

#include "matrix.h"

namespace matrix {

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(const MatrixShape<N> &s, const DataType n)
    : shape(s),
      stride(s[N - 1]) {
#ifdef DEBUG
  std::cout<<"cons" << this << std::endl;
#endif
  capicity = get_length(s, s[N - 1]);
  data = new DataType[capicity];
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
  try {
    std::uninitialized_fill(data, data + capicity, n);
  } catch (...) {
    delete[] data;
    throw;
  }
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(const MatrixShape<N> &s, const size_t st,
                                   const DataType n)
    : shape(s),
      stride(st) {
#ifdef DEBUG
  std::cout<<"cons" << this << std::endl;
#endif
  capicity = get_length(s, st);
  data = new DataType[capicity];
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
  try {
    std::uninitialized_fill(data, data + capicity, n);
  } catch (...) {
    delete[] data;
    throw;
  }
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(const Matrix<DataType, N> &m)
    : shape(m.shape),
      stride(m.stride),
      row(m.row),
      column(m.column) {
  capicity = get_length(m.shape, m.stride);
  data = new DataType[capicity];
  //std::cout<<"copy cons" << this << std::endl;
  try {
    std::uninitialized_copy(m.data, m.data + capicity, data);
  } catch (...) {
    delete[] data;
    throw;
  }
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(const SubMatrix<DataType, N> &m)
    : shape(m.shape),
      stride(m.stride),
      row(m.row),
      column(m.column) {
  capicity = get_length(m.shape, m.stride);
  data = new DataType[capicity];
  //std::cout<<"copy cons" << this << std::endl;
  try {
    std::uninitialized_copy(m.data, m.data + capicity, data);
  } catch (...) {
    delete[] data;
    throw;
  }
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(Matrix<DataType, N> &&m) {
#ifdef DEBUG
  std::cout<<"move cons" << this << std::endl;
#endif
  //get other's resource
  shape = m.shape;
  data = m.data;
  capicity = m.capicity;
  stride = m.stride;
  row = m.row;
  column = m.column;
  m.data = nullptr;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator =(
    const Matrix<DataType, N> &m) {
#ifdef DEBUG
  std::cout<<"copy assign:::" << std::endl;
#endif
  if (this != &m) {
    //free existing resource
    delete[] data;
    shape = m.shape;
    capicity = get_length(m.shape, m.stride);
    data = new DataType[capicity];
    stride = m.stride;
    row = m.row;
    column = m.column;
    try {
      std::uninitialized_copy(m.data, m.data + capicity, data);
    } catch (...) {
      delete[] data;
      throw;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator =(
    const SubMatrix<DataType, N> &m) {
#ifdef DEBUG
  std::cout<<"copy assign:::" << std::endl;
#endif
  if (this != &m) {
    //free existing resource
    delete[] data;
    shape = m.shape;
    capicity = get_length(m.shape, m.stride);
    data = new DataType[capicity];
    stride = m.stride;
    row = m.row;
    column = m.column;
    try {
      std::uninitialized_copy(m.data, m.data + capicity, data);
    } catch (...) {
      delete[] data;
      throw;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator =(
    Matrix<DataType, N> &&m) {
#ifdef DEBUG
  std::cout<<"move assign:::" << std::endl;
#endif
  if (this != &m) {
    //free existing resource
    delete[] data;
    //copy data
    shape = m.shape;
    data = m.data;
    stride = m.stride;
    capicity = m.capicity;
    row = m.row;
    column = m.column;
    //release
    m.data = nullptr;
  }
  return *this;
}
template<typename DataType, size_t N>
inline int Matrix<DataType, N>::get_capicity() const {
  size_t len = stride;
  for (int i = 0; i < N - 1; ++i) {
    len *= shape[i];
  }
  return len;
}

template<typename DataType, size_t N>
inline int Matrix<DataType, N>::get_size() const {
  size_t len = 1;
  for (int i = 0; i < N; ++i) {
    len *= shape[i];
  }
  return len;
}

template<typename DataType, size_t N>
inline size_t Matrix<DataType, N>::get_length(const MatrixShape<N> &s,
                                              size_t st) const {
  size_t len = st;
  for (int i = 0; i < N - 1; ++i) {
    len *= s.shape[i];
  }
  return len;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N - 1> Matrix<DataType, N>::operator[](
    size_t i) const {
  return SubMatrix<DataType, N - 1>(shape.subShape(),
                                    data + i * shape.stride[0], stride);  //TODO
}
template<typename DataType, size_t N>
inline SubMatrix<DataType, N - 1> Matrix<DataType, N>::operator[](size_t i) {
  return SubMatrix<DataType, N - 1>(shape.subShape(),
                                    data + i * shape.stride[0], stride);
}
template<typename DataType, size_t N>
inline SubMatrix<DataType, N> Matrix<DataType, N>::slice(size_t i,
                                                         size_t j) const {
  MatrixShape<N> s(shape);
  s[0] = j - i;
  return SubMatrix<DataType, N>(s, data + i * shape.stride[0], stride);
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> Matrix<DataType, N>::slice(size_t i, size_t j) {
  MatrixShape<N> s(shape);
  s[0] = j - i;
  return SubMatrix<DataType, N>(s, data + i * shape.stride[0], stride);
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator +=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] += n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator -=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] -= n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator *=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] *= n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator /=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] /= n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator +=(
    const Matrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] += t[i][j];
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator -=(
    const Matrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] -= t[i][j];
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator *=(
    const Matrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] *= t[i][j];
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>& Matrix<DataType, N>::operator /=(
    const Matrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] /= t[i][j];
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline DataType Matrix<DataType, N>::eval(size_t i, size_t j) const {
  return data[i * stride + j];
}

template<typename DataType, size_t N>
inline void Matrix<DataType, N>::set_row_ele(int i,
                                             const Matrix<DataType, N> & s) {
  int row_s = s.row;
  int col_s = s.column;
  if (row_s == 1) {
    if (column == col_s) {
      for (int j = 0; j < column; ++j) {
        data[i * stride + j] = s[0][j];
      }
    } else {
      std::cerr << "Set Row: Shape not match!" << std::endl;
    }
  } else {
    std::cerr << "Set Row: Shape not match!" << std::endl;
  }
}

template<typename DataType, size_t N>
template<typename SubType>
inline Matrix<DataType, N>& Matrix<DataType, N>::operator=(
    const ExprBase<SubType, DataType> &e) {
#ifdef DEBUG
  std::cout << "ET copy:" <<std::endl;
#endif
  const SubType & sub = e.self();
  ShapeCheck<SubType, N>::check(sub);
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] = sub.eval(i, j);
    }
  }
  return *this;
}

template<size_t N, typename List, typename L>
typename std::enable_if<(N == 1), void>::type fill_shape(const List & t,
                                                         L * shape) {
  *shape = t.size();
  //std::cout << shape << "&&&" << *shape<< std::endl;
  //return true;
}

template<size_t N, typename List, typename L>
typename std::enable_if<(N > 1), void>::type fill_shape(const List & t,
                                                        L * shape) {
  auto i = t.begin();
  for (auto j = i + 1; j != t.end(); ++j) {
    if (i->size() != j->size()) {
      std::cerr << "Row size inconsistent" << std::endl;
      exit(1);
    }
  }
  *shape++ = t.size();
  fill_shape<N - 1>(*t.begin(), shape);
  //return true;
}

template<typename List, size_t N>
void init_shape(const List &l, MatrixShape<N> &ss) {
  //std::array<size_t, N> a;
  auto s = ss.shape;
  fill_shape<N>(l, s);
  size_t size = 1;
  for (int i = 0; i < N; ++i) {
    size *= s[i];
  }
  shape2Stride(ss.stride, ss.shape, N);
  ss.start = 0;
}

template<typename T, typename DataType>
void fill_ele(const T* s, const T* e, DataType * data, size_t &offset) {
  for (auto i = s; i != e; ++i) {
    data[offset++] = *i;
  }
}

template<typename T, typename DataType>
void fill_ele(const std::initializer_list<T>* s,
              const std::initializer_list<T>* e, DataType * data,
              size_t &offset) {
  for (; s != e; ++s) {
    fill_ele(s->begin(), s->end(), data, offset);
  }
}

template<typename T, typename DataType>
void init_ele(std::initializer_list<T> l, DataType * d, size_t & offset) {
  fill_ele(l.begin(), l.end(), d, offset);
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(matrix_initializer_list<DataType, N> t) {
  init_shape(t, shape);
  //shape = s;
  stride = shape[N - 1];
  capicity = get_capicity();
  data = new DataType[capicity];
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
  size_t offset = 0;
  init_ele(t, data, offset);
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator =(
    matrix_initializer_list<DataType, N> t) {
  init_shape(t, shape);
  //shape = s;
  stride = shape[N - 1];
  capicity = get_capicity();
  data = new DataType[capicity];
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
  size_t offset = 0;
  init_ele(t, data, offset);
  return *this;
}

//===========================================================
//one-dimension matrix implementation
template<typename DataType>
inline Matrix<DataType, 1>::Matrix(const MatrixShape<1> &s, const DataType n)
    : shape(s),
      stride(s[0]) {
#ifdef DEBUG
  std::cout<<"cons" << this << std::endl;
#endif
  capicity = get_length(s, s[0]);
  data = new DataType[capicity];
  row = 1;
  column = shape[0];
  try {
    std::uninitialized_fill(data, data + capicity, n);
  } catch (...) {
    delete[] data;
    throw;
  }
}

template<typename DataType>
inline Matrix<DataType, 1>::Matrix(const MatrixShape<1> &s, const size_t st,
                                   const DataType n)
    : shape(s),
      stride(st) {
#ifdef DEBUG
  std::cout<<"cons" << this << std::endl;
#endif
  capicity = get_length(s, st);
  data = new DataType[capicity];
  row = 1;
  column = shape[0];
  try {
    std::uninitialized_fill(data, data + capicity, n);
  } catch (...) {
    delete[] data;
    throw;
  }
}

template<typename DataType>
inline Matrix<DataType, 1>::Matrix(const Matrix<DataType, 1> &m)
    : shape(m.shape),
      stride(m.stride),
      row(m.row),
      column(m.column) {
  capicity = get_length(m.shape, m.stride);
  data = new DataType[capicity];
  //std::cout<<"copy cons" << this << std::endl;
  try {
    std::uninitialized_copy(m.data, m.data + capicity, data);
  } catch (...) {
    delete[] data;
    throw;
  }
}

template<typename DataType>
inline Matrix<DataType, 1>::Matrix(const SubMatrix<DataType, 1> &m)
    : shape(m.shape),
      stride(m.stride),
      row(m.row),
      column(m.column) {
  capicity = get_length(m.shape, m.stride);
  data = new DataType[capicity];
  //std::cout<<"copy cons" << this << std::endl;
  try {
    std::uninitialized_copy(m.data, m.data + capicity, data);
  } catch (...) {
    delete[] data;
    throw;
  }
}

template<typename DataType>
inline Matrix<DataType, 1>::Matrix(Matrix<DataType, 1> &&m) {
#ifdef DEBUG
  std::cout<<"move cons" << this << std::endl;
#endif
  //get other's resource
  shape = m.shape;
  data = m.data;
  capicity = m.capicity;
  stride = m.stride;
  row = m.row;
  column = m.column;
  m.data = nullptr;
  // exit(0);
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator =(
    const Matrix<DataType, 1> &m) {
#ifdef DEBUG
  std::cout<<"copy assign:::" << std::endl;
#endif
  if (this != &m) {
    //free existing resource
    delete[] data;
    shape = m.shape;
    capicity = get_length(m.shape, m.stride);
    data = new DataType[capicity];
    stride = m.stride;
    row = m.row;
    column = m.column;
    try {
      std::uninitialized_copy(m.data, m.data + capicity, data);
    } catch (...) {
      delete[] data;
      throw;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator =(
    const SubMatrix<DataType, 1> &m) {
#ifdef DEBUG
  std::cout<<"copy assign:::" << std::endl;
#endif
  if (this != &m) {
    //free existing resource
    delete[] data;
    shape = m.shape;
    capicity = get_length(m.shape, m.stride);
    data = new DataType[capicity];
    stride = m.stride;
    row = m.row;
    column = m.column;
    try {
      std::uninitialized_copy(m.data, m.data + capicity, data);
    } catch (...) {
      delete[] data;
      throw;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator =(
    Matrix<DataType, 1> &&m) {
#ifdef DEBUG
  std::cout<<"move assign:::" << std::endl;
#endif
  if (this != &m) {
    //free existing resource
    delete[] data;
    //copy data
    shape = m.shape;
    data = m.data;
    stride = m.stride;
    capicity = m.capicity;
    row = m.row;
    column = m.column;
    //release
    m.data = nullptr;
  }
  return *this;
}
template<typename DataType>
inline int Matrix<DataType, 1>::get_capicity() const {
  return stride;
}

template<typename DataType>
inline int Matrix<DataType, 1>::get_size() const {
  return shape[0];
}

template<typename DataType>
inline size_t Matrix<DataType, 1>::get_length(const MatrixShape<1> &s,
                                              size_t stride) const {
  return stride;
}

template<typename DataType>
inline const DataType & Matrix<DataType, 1>::operator[](size_t i) const {
  return data[i];  //TODO
}
template<typename DataType>
inline DataType & Matrix<DataType, 1>::operator[](size_t i) {
  return data[i];
}
template<typename DataType>
inline SubMatrix<DataType, 1> Matrix<DataType, 1>::slice(size_t i,
                                                         size_t j) const {
  MatrixShape<1> s(shape);
  shape[0] = j - i;
  return SubMatrix<DataType, 1>(s, data + i, stride);
}

template<typename DataType>
inline SubMatrix<DataType, 1> Matrix<DataType, 1>::slice(size_t i, size_t j) {
  MatrixShape<1> s(shape);
  shape[0] = j - i;
  return SubMatrix<DataType, 1>(s, data + i, stride);
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator +=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] += n;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator -=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] -= n;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator *=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] *= n;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator /=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] /= n;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator +=(
    const Matrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] += t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator -=(
    const Matrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] -= t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator *=(
    const Matrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] *= t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1>& Matrix<DataType, 1>::operator /=(
    const Matrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] /= t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline DataType Matrix<DataType, 1>::eval(size_t i, size_t j) const { /*broadcasting the matrix*/
  return data[i * stride + j];
}

template<typename DataType>
inline void Matrix<DataType, 1>::set_row_ele(int i,
                                             const Matrix<DataType, 1> & s) {
  int row_s = s.row;
  int col_s = s.column;
  if (row_s == 1) {
    if (column == col_s) {
      for (int j = 0; j < column; ++j) {
        data[i * stride + j] = s[0][j];
      }
    } else {
      std::cerr << "Set Row: Shape not match!" << std::endl;
    }
  } else {
    std::cerr << "Set Row: Shape not match!" << std::endl;
  }
}

template<typename DataType>
template<typename SubType>
inline Matrix<DataType, 1>& Matrix<DataType, 1>::operator=(
    const ExprBase<SubType, DataType> &e) {
#ifdef DEBUG
  std::cout << "ET copy:" <<std::endl;
#endif
  const SubType & sub = e.self();
//ShapeCheck<SubType, 1>::check(sub);
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] = sub.eval(i, j);
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1>::Matrix(matrix_initializer_list<DataType, 1> t) {
  init_shape(t, shape);
  //shape = s;
  stride = shape[1 - 1];
  capicity = get_capicity();
  data = new DataType[capicity];
  row = 1;
  for (int i = 0; i < 1 - 1; ++i) {
    row *= shape[i];
  }
  column = shape[1 - 1];
  size_t offset = 0;
  init_ele(t, data, offset);
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator =(
    matrix_initializer_list<DataType, 1> t) {
  init_shape(t, shape);
  //shape = s;
  stride = shape[1 - 1];
  capicity = get_capicity();
  data = new DataType[capicity];
  row = 1;
  for (int i = 0; i < 1 - 1; ++i) {
    row *= shape[i];
  }
  column = shape[1 - 1];
  size_t offset = 0;
  init_ele(t, data, offset);
  return *this;
}

template<typename DataType>
std::ostream & operator <<(std::ostream &os, const Matrix<DataType, 2> & m) {
  size_t row = m.getRow();
  size_t col = m.getColumn();
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      os << m[i][j] << " ";
    }
    os << std::endl;
  }
  return os;
}

template<typename DataType>
std::ostream & operator <<(std::ostream &os, const Matrix<DataType, 1> & m) {
  size_t row = m.getRow();
  size_t col = m.getColumn();
  for (size_t j = 0; j < col; ++j) {
    os << m[j] << " ";
  }
  os << std::endl;
  return os;
}

template<typename DataType, typename StreamType>
StreamType & operator >>(StreamType &is, Matrix<DataType, 2> & m) {
  for (size_t i = 0; i < m.getRow(); ++i) {
    for (size_t j = 0; j < m.getColumn(); ++j) {
      is >> m[i][j];
    }
  }
  return is;
}

template<typename DataType>
std::ostream & operator <<(std::ostream &os, const SubMatrix<DataType, 2> & m) {
  size_t row = m.getRow();
  size_t col = m.getColumn();
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      os << m[i][j] << " ";
    }
    os << std::endl;
  }
  return os;
}

template<typename DataType, typename StreamType>
StreamType & operator >>(StreamType &is, SubMatrix<DataType, 2> & m) {
  for (size_t i = 0; i < m.getRow(); ++i) {
    for (size_t j = 0; j < m.getColumn(); ++j) {
      is >> m[i][j];
    }
  }
  return is;
}
}
#endif /* MATRIX_INL_H_ */
