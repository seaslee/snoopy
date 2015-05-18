# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export CFLAGS =  -g -O3 -std=c++11 -I../
export LFLAGS = -lopenblas

BIN =  test
.PHONY: clean all

all: $(BIN) $(OBJ)
test:  test_ml.cpp

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LFLAGS)  

clean:
	$(RM) $(OBJ) $(BIN) *~
