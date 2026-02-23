#pragma once

#include "FillStorageOps.hpp"
#define FOR_EACH_FILLSTORAGE_OP(X)  \
    X(FillOpBase)                   \
    X(FillConst)                    \
    X(FillBuffer)                   \
    X(FillArange)                   \
    X(FillLinspace)                 \
    X(FillRandomUniform)            \
    X(FillRandomNormal)             


#include "BinaryOps.hpp"
#define FOR_EACH_BINARY_OP(X)   \
    X(BinaryOpBase)             \
    X(BinaryAdd)                \
    X(BinaryAddBackward)        \
    X(BinarySub)                \
    X(BinaryExp)                 


#define FOR_EACH_OP(X)          \
    FOR_EACH_FILLSTORAGE_OP(X)  \
    FOR_EACH_BINARY_OP(X)       \