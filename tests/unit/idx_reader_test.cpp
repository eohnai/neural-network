#include "idx_reader.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {
void writeBigEndianUInt32(std::ofstream &file, std::uint32_t value) {
    const char bytes[4] = {
        static_cast<char>((value >> 24) & 0xFF),
        static_cast<char>((value >> 16) & 0xFF),
        static_cast<char>((value >> 8) & 0xFF),
        static_cast<char>(value & 0xFF),
    };

    file.write(bytes, sizeof(bytes));
}

class TemporaryIdxFile {
public:
    explicit TemporaryIdxFile(const std::string &fileName)
        : path(std::filesystem::temp_directory_path() / fileName) {
        std::filesystem::remove(path);
    }

    ~TemporaryIdxFile() {
        std::filesystem::remove(path);
    }

    std::filesystem::path path;
};
} // namespace

TEST(IdxReaderTest, ReadsImageHeaderAndPixels) {
    TemporaryIdxFile testFile("idx_reader_image_test.idx");

    {
        std::ofstream file(testFile.path, std::ios::binary);
        ASSERT_TRUE(file.is_open());

        writeBigEndianUInt32(file, 0x00000803); // Unsigned byte values, 3 dimensions.
        writeBigEndianUInt32(file, 2);          // Two images.
        writeBigEndianUInt32(file, 2);          // Two rows per image.
        writeBigEndianUInt32(file, 2);          // Two columns per image.

        const std::uint8_t pixels[] = {0, 1, 2, 3, 4, 5, 6, 255};
        file.write(reinterpret_cast<const char *>(pixels), sizeof(pixels));
    }

    const IdxFile idxFile = IdxReader::read(testFile.path.string());

    EXPECT_EQ(idxFile.idxType, 0x08);
    EXPECT_EQ(idxFile.dimensions, (std::vector<std::uint32_t>{2, 2, 2}));
    EXPECT_EQ(idxFile.data,
              (std::vector<std::uint8_t>{0, 1, 2, 3, 4, 5, 6, 255}));
}

TEST(IdxReaderTest, ReadsLabelHeaderAndLabels) {
    TemporaryIdxFile testFile("idx_reader_label_test.idx");

    {
        std::ofstream file(testFile.path, std::ios::binary);
        ASSERT_TRUE(file.is_open());

        writeBigEndianUInt32(file, 0x00000801); // Unsigned byte values, 1 dimension.
        writeBigEndianUInt32(file, 3);          // Three labels.

        const std::uint8_t labels[] = {7, 0, 9};
        file.write(reinterpret_cast<const char *>(labels), sizeof(labels));
    }

    const IdxFile idxFile = IdxReader::read(testFile.path.string());

    EXPECT_EQ(idxFile.idxType, 0x08);
    EXPECT_EQ(idxFile.dimensions, (std::vector<std::uint32_t>{3}));
    EXPECT_EQ(idxFile.data, (std::vector<std::uint8_t>{7, 0, 9}));
}

TEST(IdxReaderTest, RejectsTruncatedPayload) {
    TemporaryIdxFile testFile("idx_reader_truncated_test.idx");

    {
        std::ofstream file(testFile.path, std::ios::binary);
        ASSERT_TRUE(file.is_open());

        writeBigEndianUInt32(file, 0x00000803);
        writeBigEndianUInt32(file, 1);
        writeBigEndianUInt32(file, 2);
        writeBigEndianUInt32(file, 2);

        const std::uint8_t incompletePixels[] = {0, 1, 2};
        file.write(reinterpret_cast<const char *>(incompletePixels), sizeof(incompletePixels));
    }

    EXPECT_THROW(IdxReader::read(testFile.path.string()), std::runtime_error);
}
