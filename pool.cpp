#include "pool.h"

void* yzw2v::mem::Pool::Align(const size_t alignment, const size_t size, void*&ptr, size_t& space) noexcept {
    // copy-pasted libc++ implementation;
    void* r = nullptr;
    if (size <= space)
    {
        char* p1 = static_cast<char*>(ptr);
        char* p2 = reinterpret_cast<char*>(reinterpret_cast<size_t>(p1 + (alignment - 1)) & -alignment);
        size_t d = static_cast<size_t>(p2 - p1);
        if (d <= space - size)
        {
            r = p2;
            ptr = r;
            space -= d;
        }
    }
    return r;
}
