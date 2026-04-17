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
    X(BinaryExp)                \
    X(BinaryMul)                \
    X(BinaryDiv)

#include "ReductionOps.hpp"
#define FOR_EACH_REDUCTION_OP(X) \
    X(ReduceSum)                 \
    X(ReduceMul)                 \
    X(ReduceMax)                 \
    X(ReduceMin)

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

#include "MatMulOps.hpp"
#define FOR_EACH_MATMUL_OP(X) \
    X(MatMul)       

#define FOR_EACH_OP(X)          \
    FOR_EACH_FILLSTORAGE_OP(X)  \
    FOR_EACH_BINARY_OP(X)       \
    FOR_EACH_REDUCTION_OP(X)    \
    FOR_EACH_UNARY_OP(X)        \
    FOR_EACH_MATMUL_OP(X)