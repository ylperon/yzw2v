#pragma once

#include "likely.h"

#include <list>
#include <memory>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace yzw2v {
    namespace mem {
        class Pool {
        public:
            explicit Pool(const size_t block_size)
                : default_block_size{block_size} {
                blocks_.emplace_back(default_block_size);
            }

            template <typename T>
            T* Get(const size_t size) {
                return reinterpret_cast<T*>(GetAligned(sizeof(T) * size, alignof(T)));
            }

        private:
            static void* Align(const size_t alignment, const size_t size, void*&ptr, size_t& space) noexcept;

            void* GetAligned(const size_t size, const size_t alignment) {
                auto* res = blocks_.back().current;
                if (Align(alignment, size, res, blocks_.back().memory_left)) {
                    blocks_.back().current = reinterpret_cast<uint8_t*>(res) + size;
                    blocks_.back().memory_left -= size;
                    return res;
                } else {
                    const auto size_to_allocate = default_block_size < size + alignment - 1
                                                    ? size + alignment - 1
                                                    : default_block_size;
                    blocks_.emplace_back(size_to_allocate);
                }


                res = Align(alignment, size,
                                 blocks_.back().current, blocks_.back().memory_left);
                if (!res) {
                    assert(false); // you was wrong
                }

                blocks_.back().current = reinterpret_cast<uint8_t*>(res) + size;
                blocks_.back().memory_left -= size;
                return res;
            }

            struct Block {
                void* current;
                size_t memory_left;
                std::unique_ptr<uint8_t[]> begin;

                explicit Block(const size_t custom_block_size)
                    : memory_left{custom_block_size}
                    , begin{new uint8_t[custom_block_size]}
                {
                    current = begin.get();
                    std::memset(current, 0, memory_left);
                }
            };

            size_t default_block_size;
            std::list<Block> blocks_;
        };
    };
}
