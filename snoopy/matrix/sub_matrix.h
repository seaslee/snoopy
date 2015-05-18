/**
 *  \file matrix.h
 *  \brief a template class to describe the SubMatrix
 *  \author wbd
 *
 *  This class implements the template submatrix for the index and slice opeartor
 *  The SubMatrix is just a clone of Matrix class except that it can't manager the
 *  memory resources of the matrix in it's constructor and deconstructor.
 *  TODO: A more elegant way to do the index and slice opeartor.
 */

#ifndef SUB_MATRIX_H_
#define SUB_MATRIX_H_

namespace matrix {

template<typename DataType, size_t N>
class SubMatrix : public ExprBase<SubMatrix<DataType, N>, DataType> {
 public:
  inline SubMatrix()
      : data(nullptr),
        stride(size_t(0)) {
  }
  ;
  inline SubMatrix(const MatrixShape<N> &s, DataType * d, const DataType n);
  inline SubMatrix(const MatrixShape<N> &s, DataType * d, const size_t st,
                   const DataType n);
  inline SubMatrix(const SubMatrix<DataType, N> &m);
  inline ~SubMatrix() {
#ifdef DEBUG
    std::cout<<"decons!!!!!!!!!!!!!!!!!!!!" << this << std::endl;
#endif
  }

  inline SubMatrix<DataType, N> & operator =(const SubMatrix<DataType, N> &m);
  inline int get_size() const;
  inline size_t get_length(const MatrixShape<N> &s, size_t stride) const;
  inline DataType * get_data() {
    return data;
  }
  inline DataType * get_data() const {
    return data;
  }
  inline SubMatrix<DataType, N - 1> operator[](size_t i) const;
  inline SubMatrix<DataType, N - 1> operator[](size_t i);
  inline SubMatrix<DataType, N> slice(size_t i, size_t j) const;
  inline SubMatrix<DataType, N> slice(size_t i, size_t j);
  inline SubMatrix<DataType, N> & operator +=(const DataType & n);
  inline SubMatrix<DataType, N> & operator -=(const DataType & n);
  inline SubMatrix<DataType, N> & operator *=(const DataType & n);
  inline SubMatrix<DataType, N> & operator /=(const DataType & n);
  SubMatrix<DataType, N>& operator +=(const SubMatrix<DataType, N>& t);
  inline SubMatrix<DataType, N> & operator -=(const SubMatrix<DataType, N> & t);
  inline SubMatrix<DataType, N> & operator *=(const SubMatrix<DataType, N> & t);
  inline SubMatrix<DataType, N>& operator /=(const SubMatrix<DataType, N> & t);
  template<typename SubType>
  inline SubMatrix<DataType, N>& operator=(
      const ExprBase<SubType, DataType> &e);
  inline DataType eval(size_t i, size_t j) const;
  size_t getColumn() const {
    return column;
  }
  size_t getRow() const {
    return row;
  }
  size_t getSize() const {
    return size;
  }
  //private:
  MatrixShape<N> shape;
  size_t stride;
  size_t size;
  DataType * data;
  size_t row;
  size_t column;
};

//
template<typename DataType>
class SubMatrix<DataType, 1> : public ExprBase<SubMatrix<DataType, 1>, DataType> {
 public:
  /*
   * default constructor
   */
  inline SubMatrix()
      : data(nullptr),
        stride(size_t(0)) {
  }
  ;
  inline SubMatrix(const MatrixShape<1> &s, DataType * d, const DataType n);
  inline SubMatrix(const MatrixShape<1> &s, DataType * d, const size_t st,
                   const DataType n);
  inline SubMatrix(const SubMatrix<DataType, 1> &m);
  inline ~SubMatrix() {
#ifdef DEBUG
    std::cout<<"decons!!!!!!!!!!!!!!!!!!!!" << this << std::endl;
#endif
  }
  inline SubMatrix<DataType, 1> & operator =(const SubMatrix<DataType, 1> &m);
  inline int get_size() const {
    return stride;
  }
  inline size_t get_length(const MatrixShape<1> &s, size_t st) const {
    return st;
  }
  inline DataType * get_data() {
    return data;
  }

  inline DataType * get_data() const {
    return data;
  }

  inline DataType& operator[](size_t i) const {
    return data[i];
  }

  inline DataType& operator[](size_t i) {
    return data[i];
  }

  inline SubMatrix<DataType, 1> slice(size_t i, size_t j) const;
  inline SubMatrix<DataType, 1> slice(size_t i, size_t j);
  inline SubMatrix<DataType, 1> & operator +=(const DataType & n);
  inline SubMatrix<DataType, 1> & operator -=(const DataType & n);
  inline SubMatrix<DataType, 1> & operator *=(const DataType & n);
  inline SubMatrix<DataType, 1> & operator /=(const DataType & n);
  inline SubMatrix<DataType, 1> & operator +=(const SubMatrix<DataType, 1> & t);
  inline SubMatrix<DataType, 1> & operator -=(const SubMatrix<DataType, 1> & t);
  inline SubMatrix<DataType, 1> & operator *=(const SubMatrix<DataType, 1> & t);
  inline SubMatrix<DataType, 1>& operator /=(const SubMatrix<DataType, 1> & t);
  template<typename SubType>
  inline SubMatrix<DataType, 1>& operator=(
      const ExprBase<SubType, DataType> &e);
  inline DataType eval(size_t i, size_t j) const;
  size_t getColumn() const {
    return column;
  }
  size_t getRow() const {
    return row;
  }
  size_t getSize() const {
    return size;
  }
  //private:
  MatrixShape<1> shape;
  size_t stride;
  size_t size;
  DataType * data;
  size_t row;
  size_t column;
};

}

#endif /* SUB_MATRIX_H_ */
