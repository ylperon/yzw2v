#if defined(__AVX__)
#pragma message "[yzw2v] will use AVX instruction set for linear algebra"
#include "numeric_avx.cpp"
#elif defined(__SSE__)
#pragma message "[yzw2v] will use SSE instruction set for linear algebra"
#include "numeric_sse.cpp"
#else
#pragma message "[yzw2v] will use basic C++ code for linear algebra"
#include "numeric_simple.cpp"
#endif
