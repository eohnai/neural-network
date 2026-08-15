#pragma once

#include "idx_reader.hpp"
#include "matrix.hpp"

#include <cstddef>
#include <string>

class MNISTDataset {
private:
    IdxFile images;
    IdxFile labels;

public:
    MNISTDataset(const std::string &imagePath, const std::string &labelPath);

    std::size_t size() const;
    Matrix imageAt(std::size_t index) const;
    Matrix inputAt(std::size_t index) const;
    Matrix targetAt(std::size_t index) const;
};
