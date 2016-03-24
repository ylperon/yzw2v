#pragma once

#if defined(__GNUC__)
#define YZ_PREFETCH_READ(ptr, priority) __builtin_prefetch(reinterpret_cast<const void*>(ptr), 0, priority)
#define YZ_PREFETCH_WRITE(ptr, priority) __builtin_prefetch(reinterpret_cast<const void*>(ptr), 1, priority)
#else
#define YZ_PREFETCH_READ(ptr, priority) (void)(Pointer), (void)Priority
#define YZ_PREFETCH_WRITE(ptr, priority) (void)(Pointer), (void)Priority
#endif
