#include "idx_reader.hpp"

#include <fstream>
#include <stdexcept>

namespace {
std::uint32_t readBigEndianUInt32(std::ifstream &file) {
    std::uint32_t value = 0;

    for (int byteIndex = 0; byteIndex < 4; ++byteIndex) {
        char byte;

        // Read the next byte into `byte` and advance the file cursor.
        // The condition checks whether that read succeeded.
        if (!file.get(byte)) {
            throw std::runtime_error("We have reached the end of the IDX file!");
        }

        // Shift earlier bytes left by one byte, then append the new byte.
        // Convert through unsigned char so raw byte values remain in the range 0-255.
        value = (value << 8) | static_cast<std::uint32_t>(static_cast<unsigned char>(byte));
    }

    return value;
}
} // namespace

namespace IdxReader {
IdxFile read(const std::string &filePath) {
    // open the file stream safely
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open the file: " + filePath);
    }

    std::uint32_t idxTypeAndDimensionCount = readBigEndianUInt32(file);

    IdxFile idxFile;
    idxFile.idxType = static_cast<std::uint8_t>((idxTypeAndDimensionCount >> 8) & 0xFF);
    const std::uint8_t numberOfDimensions =
        static_cast<std::uint8_t>(idxTypeAndDimensionCount & 0xFF);

    if (idxFile.idxType != 0x08) {
        throw std::runtime_error("Only unsigned-byte IDX files are supported.");
    }

    if (numberOfDimensions == 0) {
        throw std::runtime_error("IDX file must declare at least one dimension.");
    }

    for (int dimension = 0; dimension < numberOfDimensions; ++dimension) {
        idxFile.dimensions.push_back(readBigEndianUInt32(file));
    }

    std::size_t elementCount = 1;
    for (const std::uint32_t dimension : idxFile.dimensions) {
        elementCount *= dimension;
    }

    // IDX type 0x08 stores one byte for every pixel or label.
    idxFile.data.resize(elementCount);
    file.read(reinterpret_cast<char *>(idxFile.data.data()),
              static_cast<std::streamsize>(idxFile.data.size()));

    if (!file) {
        throw std::runtime_error("IDX file ended before all declared data could be read.");
    }

    return idxFile;
}
} // namespace IdxReader
