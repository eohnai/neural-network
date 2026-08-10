#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct IdxFile {
    std::uint8_t idxType;
    std::vector<std::uint32_t> dimensions;
    std::vector<std::uint8_t> data;
};

namespace IdxReader {
IdxFile read(const std::string &filePath);
} // namespace IdxReader
