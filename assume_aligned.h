#pragma once

#if defined(__GNUC__)
#define YZ_ASSUME_ALIGNED(ptr, alignment) reinterpret_cast<decltype(ptr)>(__builtin_assume_aligned(ptr, alignment))
#else
#define YZ_ASSUME_ALIGNED(ptr, alignment) ptr
#endif
