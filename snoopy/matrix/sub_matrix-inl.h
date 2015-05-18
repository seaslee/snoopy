#ifndef SUB_MATRIX_INL_H_
#define SUB_MATRIX_INL_H_

#include "sub_matrix.h"

namespace matrix {
template<typename DataType, size_t N>
inline SubMatrix<DataType, N>::SubMatrix(const MatrixShape<N> &s, DataType * d,
                                         const DataType n)
    : shape(s),
      stride(s[N - 1]) {
#ifdef DEBUG
  std::cout<<"cons" << this << std::endl;
#endif
  size = get_length(s, s[N - 1]);
  data = d;
  row = 1;
  for (int i = 0; i < 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N>::SubMatrix(const MatrixShape<N> &s, DataType * d,
                                         const size_t st, const DataType n)
    : shape(s),
      stride(st) {
#ifdef DEBUG
  std::cout<<"cons" << this << std::endl;
#endif
  size = get_length(s, st);
  data = d;
  row = 1;
  for (int i = 0; i < 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N>::SubMatrix(const SubMatrix<DataType, N> &m)
    : shape(m.shape),
      stride(m.stride),
      row(m.row),
      column(m.column) {
  size = get_length(m.shape, m.stride);
  data = m.data;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> & SubMatrix<DataType, N>::operator =(
    const SubMatrix<DataType, N> &m) {
#ifdef DEBUG
  std::cout<<"copy assign:::" << std::endl;
#endif
  shape = m.shape;
  size = get_length(m.shape, m.stride);
  data = m.data;
  stride = m.stride;
  row = m.row;
  column = m.column;
  return *this;
}

template<typename DataType, size_t N>
inline int SubMatrix<DataType, N>::get_size() const {
  size_t len = stride;
  for (int i = 0; i < N - 1; ++i) {
    len *= shape[i];
  }
  return len;
}

template<typename DataType, size_t N>
inline size_t SubMatrix<DataType, N>::get_length(const MatrixShape<N> &s,
                                                 size_t stride) const {
  size_t len = stride;
  for (int i = 0; i < N - 1; ++i) {
    len *= shape[i];
  }
  return len;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N - 1> SubMatrix<DataType, N>::operator[](
    size_t i) const {
  return SubMatrix<DataType, N - 1>(shape.subShape(),
                                    data + i * shape.stride[0], stride);  //TODO
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N - 1> SubMatrix<DataType, N>::operator[](size_t i) {
  return SubMatrix<DataType, N - 1>(shape.subShape(),
                                    data + i * shape.stride[0], stride);
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> SubMatrix<DataType, N>::slice(size_t i,
                                                            size_t j) const {
  MatrixShape<N> s(shape);
  shape[0] = j - i;
  return SubMatrix<DataType, N>(s, data + i * shape.stride[0], stride);
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> SubMatrix<DataType, N>::slice(size_t i,
                                                            size_t j) {
  MatrixShape<N> s(shape);
  shape[0] = j - i;
  return SubMatrix<DataType, N>(s, data + i * shape.stride[0], stride);
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> & SubMatrix<DataType, N>::operator +=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] += n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> & SubMatrix<DataType, N>::operator -=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] -= n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> & SubMatrix<DataType, N>::operator *=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] *= n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> & SubMatrix<DataType, N>::operator /=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] /= n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N>& SubMatrix<DataType, N>::operator +=(
    const SubMatrix<DataType, N>& t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] += t[i][j];
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> & SubMatrix<DataType, N>::operator -=(
    const SubMatrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] -= t[i][j];
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N> & SubMatrix<DataType, N>::operator *=(
    const SubMatrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] *= t[i][j];
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline SubMatrix<DataType, N>& SubMatrix<DataType, N>::operator /=(
    const SubMatrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] /= t[i][j];
    }
  }
  return *this;
}

template<typename DataType, size_t N>
template<typename SubType>
inline SubMatrix<DataType, N>& SubMatrix<DataType, N>::operator=(
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

template<typename DataType, size_t N>
inline DataType SubMatrix<DataType, N>::eval(size_t i, size_t j) const { /*broadcasting the matrix*/
  return data[i * stride + j];
}

//========================================================
template<typename DataType>
inline SubMatrix<DataType, 1>::SubMatrix(const MatrixShape<1> &s, DataType * d,
                                         const DataType n)
    : shape(s),
      stride(s[0]) {
#ifdef DEBUG
  std::cout<<"cons" << this << std::endl;
#endif
  size = get_length(s, s[0]);
  data = d;
  row = 1;
  column = shape[0];
}

template<typename DataType>
inline SubMatrix<DataType, 1>::SubMatrix(const MatrixShape<1> &s, DataType * d,
                                         const size_t st, const DataType n)
    : shape(s),
      stride(st) {
#ifdef DEBUG
  std::cout<<"cons" << this << std::endl;
#endif
  size = get_length(s, st);
  data = d;
  row = 1;
  column = shape[0];
}

template<typename DataType>
inline SubMatrix<DataType, 1>::SubMatrix(const SubMatrix<DataType, 1> &m)
    : shape(m.shape),
      stride(m.stride),
      row(m.row),
      column(m.column) {
  size = get_length(m.shape, m.stride);
  data = m.data;
}

template<typename DataType>
inline SubMatrix<DataType, 1> & SubMatrix<DataType, 1>::operator =(
    const SubMatrix<DataType, 1> &m) {
#ifdef DEBUG
  std::cout<<"copy assign:::" << std::endl;
#endif
  shape = m.shape;
  size = get_length(m.shape, m.stride);
  data = m.d;
  stride = m.stride;
  row = m.row;
  column = m.column;
  return *this;
}

template<typename DataType>
inline SubMatrix<DataType, 1> SubMatrix<DataType, 1>::slice(size_t i,
                                                            size_t j) const {
  MatrixShape<1> s(shape);
  shape[0] = j - i;
  return SubMatrix<DataType, 1>(s, data + i * shape.stride[0], stride);
}
template<typename DataType>
inline SubMatrix<DataType, 1> SubMatrix<DataType, 1>::slice(size_t i,
                                                            size_t j) {
  MatrixShape<1> s(shape);
  shape[0] = j - i;
  return SubMatrix<DataType, 1>(s, data + i * shape.stride[0], stride);
}

template<typename DataType>
inline SubMatrix<DataType, 1> & SubMatrix<DataType, 1>::operator +=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] += n;
    }
  }
  return *this;
}

template<typename DataType>
inline SubMatrix<DataType, 1> & SubMatrix<DataType, 1>::operator -=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] -= n;
    }
  }
  return *this;
}

template<typename DataType>
inline SubMatrix<DataType, 1> & SubMatrix<DataType, 1>::operator *=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] *= n;
    }
  }
  return *this;
}

template<typename DataType>
inline SubMatrix<DataType, 1> & SubMatrix<DataType, 1>::operator /=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] /= n;
    }
  }
  return *this;
}

template<typename DataType>
inline SubMatrix<DataType, 1> & SubMatrix<DataType, 1>::operator +=(
    const SubMatrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] += t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline SubMatrix<DataType, 1> & SubMatrix<DataType, 1>::operator -=(
    const SubMatrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] -= t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline SubMatrix<DataType, 1> & SubMatrix<DataType, 1>::operator *=(
    const SubMatrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] *= t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline SubMatrix<DataType, 1>& SubMatrix<DataType, 1>::operator /=(
    const SubMatrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] /= t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
template<typename SubType>
inline SubMatrix<DataType, 1>& SubMatrix<DataType, 1>::operator=(
    const ExprBase<SubType, DataType> &e) {
#ifdef DEBUG
  std::cout << "ET copy:" <<std::endl;
#endif
  const SubType & sub = e.self();
  ShapeCheck<SubType, 1>::check(sub);
  //std::tuple<int,int> t = sub.check();
  //check
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data[i * stride + j] = sub.eval(i, j);
    }
  }
  return *this;
}

template<typename DataType>
inline DataType SubMatrix<DataType, 1>::eval(size_t i, size_t j) const {
  return data[i * stride + j];
}

}
#endif /* SUB_MATRIX_INL_H_ */
