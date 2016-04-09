#pragma once

#if defined(__GNUC__)
#define YZ_ASSUME(exr) __builtin_assume((exr))
#else
#define YZ_ASSUME(exr) (void)(exp)
#endif
