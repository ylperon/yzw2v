#pragma once

#if defined(__GNUC__) && __GNUC__ >= 4
#define YZ_LIKELY(x)   (__builtin_expect((x), 1))
#define YZ_UNLIKELY(x) (__builtin_expect((x), 0))
#else
#define YZ_LIKELY(x)   (x)
#define YZ_UNLIKELY(x) (x)
#endif
