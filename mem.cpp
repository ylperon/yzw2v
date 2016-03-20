#if defined(__linux__) || defined(__APPLE__)
#include "mem_posix.cpp"
#else
#error "No implementation for current platform"
#endif
