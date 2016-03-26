#pragma once

#if defined(__GNUC__)
#define YZ_PREFETCH_READ(ptr, priority) __builtin_prefetch(ptr, 0, priority)
#define YZ_PREFETCH_WRITE(ptr, priority) __builtin_prefetch(ptr, 1, priority)
#else
#define YZ_PREFETCH_READ(ptr, priority) (void)(pointer), (void)priority
#define YZ_PREFETCH_WRITE(ptr, priority) (void)(pointer), (void)priority
#endif
