#include "mnist_dataset.hpp"

#include <stdexcept>

MNISTDataset::MNISTDataset(const std::string &imagePath, const std::string &labelPath)
    : images(IdxReader::read(imagePath)), labels(IdxReader::read(labelPath)) {
    if (images.dimensions.size() != 3 || images.dimensions[1] != 28 || images.dimensions[2] != 28) {
        throw std::invalid_argument("MNIST images must have dimensions [count, 28, 28].");
    }

    if (labels.dimensions.size() != 1) {
        throw std::invalid_argument("MNIST labels must have dimensions [count].");
    }

    if (images.dimensions[0] != labels.dimensions[0]) {
        throw std::invalid_argument("MNIST images and labels have different sample counts.");
    }
}

std::size_t MNISTDataset::size() const { return images.dimensions[0]; }

Matrix MNISTDataset::imageAt(std::size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("MNIST image index is out of range.");
    }

    constexpr int imageRows = 28;
    constexpr int imageColumns = 28;
    constexpr std::size_t pixelsPerImage = imageRows * imageColumns;
    const std::size_t imageOffset = index * pixelsPerImage;

    Matrix image(imageRows, imageColumns);

    for (int row = 0; row < imageRows; ++row) {
        for (int column = 0; column < imageColumns; ++column) {
            const std::size_t pixelIndex = row * imageColumns + column;
            const std::uint8_t rawPixel = images.data[imageOffset + pixelIndex];
            image(row, column) = static_cast<double>(rawPixel) / 255.0;
        }
    }

    return image;
}

Matrix MNISTDataset::inputAt(std::size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("MNIST image index is out of range.");
    }

    constexpr std::size_t pixelsPerImage = 28 * 28;
    const std::size_t imageOffset = index * pixelsPerImage;

    Matrix input(static_cast<int>(pixelsPerImage), 1);

    for (std::size_t pixelIndex = 0; pixelIndex < pixelsPerImage; ++pixelIndex) {
        const std::uint8_t rawPixel = images.data[imageOffset + pixelIndex];
        input(static_cast<int>(pixelIndex), 0) = static_cast<double>(rawPixel) / 255.0;
    }

    return input;
}

Matrix MNISTDataset::targetAt(std::size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("MNIST label index is out of range.");
    }

    const std::uint8_t label = labels.data[index];

    if (label >= 10) {
        throw std::runtime_error("MNIST label must be between 0 and 9.");
    }

    Matrix target(10, 1);
    target(static_cast<int>(label), 0) = 1.0;

    return target;
}
