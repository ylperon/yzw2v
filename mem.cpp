#if defined(__linux__) || defined(__APPLE__)
#include "mem_posix.cpp"
#elif defined(_WIN32) || defined(_WIN64)
#include "mem_win.cpp"
#else
#error "No implementation for current platform"
#endif
