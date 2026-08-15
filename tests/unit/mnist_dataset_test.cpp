#include "mnist_dataset.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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

void writeUnsignedByteIdxFile(const std::filesystem::path &path,
                              const std::vector<std::uint32_t> &dimensions,
                              const std::vector<std::uint8_t> &data) {
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open());

    const std::uint32_t typeAndDimensionCount =
        0x00000800 | static_cast<std::uint32_t>(dimensions.size());
    writeBigEndianUInt32(file, typeAndDimensionCount);

    for (const std::uint32_t dimension : dimensions) {
        writeBigEndianUInt32(file, dimension);
    }

    file.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size()));
}

class TemporaryMnistFiles {
  public:
    TemporaryMnistFiles()
        : imagePath(std::filesystem::temp_directory_path() / "mnist_dataset_images_test.idx"),
          labelPath(std::filesystem::temp_directory_path() / "mnist_dataset_labels_test.idx") {
        std::filesystem::remove(imagePath);
        std::filesystem::remove(labelPath);
    }

    ~TemporaryMnistFiles() {
        std::filesystem::remove(imagePath);
        std::filesystem::remove(labelPath);
    }

    std::filesystem::path imagePath;
    std::filesystem::path labelPath;
};
} // namespace

TEST(MnistDatasetTest, ConvertsImagesAndLabelsToTrainingMatrices) {
    TemporaryMnistFiles testFiles;

    std::vector<std::uint8_t> pixels(2 * 28 * 28, 0);
    pixels[1] = 255;
    pixels[783] = 127;
    pixels[784] = 64;

    writeUnsignedByteIdxFile(testFiles.imagePath, {2, 28, 28}, pixels);
    writeUnsignedByteIdxFile(testFiles.labelPath, {3}, {3, 9, 1});

    EXPECT_THROW(MNISTDataset(testFiles.imagePath.string(), testFiles.labelPath.string()),
                 std::invalid_argument);

    writeUnsignedByteIdxFile(testFiles.labelPath, {2}, {3, 9});
    MNISTDataset dataset(testFiles.imagePath.string(), testFiles.labelPath.string());

    EXPECT_EQ(dataset.size(), 2U);

    const Matrix firstImage = dataset.imageAt(0);
    EXPECT_EQ(firstImage.getRows(), 28);
    EXPECT_EQ(firstImage.getCols(), 28);
    EXPECT_DOUBLE_EQ(firstImage(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(firstImage(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(firstImage(27, 27), 127.0 / 255.0);

    const Matrix secondImage = dataset.imageAt(1);
    EXPECT_DOUBLE_EQ(secondImage(0, 0), 64.0 / 255.0);

    const Matrix firstInput = dataset.inputAt(0);
    EXPECT_EQ(firstInput.getRows(), 784);
    EXPECT_EQ(firstInput.getCols(), 1);
    EXPECT_DOUBLE_EQ(firstInput(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(firstInput(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(firstInput(783, 0), 127.0 / 255.0);

    const Matrix secondInput = dataset.inputAt(1);
    EXPECT_DOUBLE_EQ(secondInput(0, 0), 64.0 / 255.0);

    const Matrix firstTarget = dataset.targetAt(0);
    EXPECT_EQ(firstTarget.getRows(), 10);
    EXPECT_EQ(firstTarget.getCols(), 1);

    for (int row = 0; row < 10; ++row) {
        EXPECT_DOUBLE_EQ(firstTarget(row, 0), row == 3 ? 1.0 : 0.0);
    }

    const Matrix secondTarget = dataset.targetAt(1);
    EXPECT_DOUBLE_EQ(secondTarget(9, 0), 1.0);
}

TEST(MnistDatasetTest, RejectsOutOfRangeSampleIndices) {
    TemporaryMnistFiles testFiles;

    writeUnsignedByteIdxFile(testFiles.imagePath, {1, 28, 28},
                             std::vector<std::uint8_t>(28 * 28, 0));
    writeUnsignedByteIdxFile(testFiles.labelPath, {1}, {0});

    MNISTDataset dataset(testFiles.imagePath.string(), testFiles.labelPath.string());

    EXPECT_THROW(dataset.imageAt(1), std::out_of_range);
    EXPECT_THROW(dataset.inputAt(1), std::out_of_range);
    EXPECT_THROW(dataset.targetAt(1), std::out_of_range);
}
