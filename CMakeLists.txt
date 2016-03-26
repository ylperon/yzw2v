cmake_minimum_required(VERSION 2.8)
project(yzw2v)

find_package(Threads REQUIRED)

set(BUILD_STATIC_LIBS ON)
set(BUILD_SHARED_LIBS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# for profiling
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")

# Compiler dependent options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-documentation-unknown-command")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-exit-time-destructors")

    if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    elseif ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
else()
    message(SEND_ERROR "${CMAKE_CXX_COMPILER_ID} is not supported")
endif()

## Build type dependent options

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto")

add_executable(yzw2v
    main.cpp
    vocabulary.cpp
    collect_vocabulary.cpp
    io_vocabulary.cpp
    io.cpp
    pool.cpp
    huffman.cpp
    token_reader.cpp
    train.cpp
    io_train.cpp
    mem.cpp
    numeric.cpp
    unigram_distribution_imprecise.cpp
    # unigram_distribution_precise.cpp
    matrix.cpp
    mem_vec_size.cpp
)

target_link_libraries(yzw2v
    ${CMAKE_THREAD_LIBS_INIT}
)
