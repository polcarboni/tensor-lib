#pragma once

#include "FillStorageOps.hpp"
#define FOR_EACH_FILLSTORAGE_OP(X)  \
    X(FillConst)                    \
    X(FillArange)                   \
    X(FillLinspace)                 \
    X(FillRandomUniform)            \
    X(FillRandomNormal)             \
    X(FillEye)

#include "BinaryOps.hpp"
#define FOR_EACH_BINARY_OP(X)   \
    X(BinaryAdd)                \
    X(BinaryAddBackward)        \
    X(BinarySub)                \
    X(BinaryExp)                 

#include "ReductionOps.hpp"
#define FOR_EACH_REDUCTION_OP(X) \
    X(ReduceSum)

#include "UnaryOps.hpp"
#define FOR_EACH_UNARY_OP(X) \
    X(UnaryCastOp)           \
    X(ContiguousOp)          \
    X(UnaryNeg)              \
    X(UnaryAbs)              \
    X(UnaryExp)              \
    X(UnaryLog)              \
    X(UnarySigmoid)          \
    X(UnaryTanh)             \
    X(UnaryRelu)             


#define FOR_EACH_OP(X)          \
    FOR_EACH_FILLSTORAGE_OP(X)  \
    FOR_EACH_BINARY_OP(X)       \
    FOR_EACH_REDUCTION_OP(X)    \
    FOR_EACH_UNARY_OP(X)
