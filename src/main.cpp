#include "mnist_dataset.hpp"
#include "neural_network.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr int inputNodes = 28 * 28;
constexpr int hiddenNodes = 128;
constexpr int outputNodes = 10;

std::size_t parsePositiveSize(std::string_view text, std::string_view argumentName) {
    std::size_t value = 0;
    const auto [position, error] = std::from_chars(text.data(), text.data() + text.size(), value);

    if (error != std::errc{} || position != text.data() + text.size() || value == 0) {
        throw std::invalid_argument(std::string(argumentName) + " must be a positive integer.");
    }

    return value;
}

std::filesystem::path defaultDataDirectory() {
#ifdef NEURAL_NETWORK_SOURCE_DIR
    return std::filesystem::path(NEURAL_NETWORK_SOURCE_DIR) / "data" / "mnist" / "raw";
#else
    return "data/mnist/raw";
#endif
}

int indexOfLargestValue(const Matrix &matrix) {
    int largestIndex = 0;

    for (int row = 1; row < matrix.getRows(); ++row) {
        if (matrix(row, 0) > matrix(largestIndex, 0)) {
            largestIndex = row;
        }
    }

    return largestIndex;
}

void printUsage(const char *programName) {
    std::cout << "Usage: " << programName << " [epochs] [max-training-samples] [data-directory]\n"
              << "Defaults: 1 epoch, all training samples, data/mnist/raw\n";
}

} // namespace

int main(int argc, char *argv[]) {
    try {
        if (argc > 4) {
            printUsage(argv[0]);
            return 1;
        }

        const std::size_t epochs = argc >= 2 ? parsePositiveSize(argv[1], "epochs") : 1;
        const std::filesystem::path dataDirectory = argc >= 4 ? argv[3] : defaultDataDirectory();

        const MNISTDataset trainingData(
            (dataDirectory / "train-images.idx3-ubyte").string(),
            (dataDirectory / "train-labels.idx1-ubyte").string());
        const MNISTDataset testData(
            (dataDirectory / "t10k-images.idx3-ubyte").string(),
            (dataDirectory / "t10k-labels.idx1-ubyte").string());

        const std::size_t requestedSamples = argc >= 3
                                                 ? parsePositiveSize(argv[2], "max-training-samples")
                                                 : trainingData.size();
        const std::size_t trainingSamples = std::min(requestedSamples, trainingData.size());

        NeuralNetwork network(inputNodes, hiddenNodes, outputNodes, ActivationType::SIGMOID, 20260815);
        std::vector<std::size_t> sampleIndices(trainingSamples);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        std::mt19937_64 shuffleGenerator(20260815);

        std::cout << "Training MNIST network: " << inputNodes << " -> " << hiddenNodes << " -> " << outputNodes
                  << "\nTraining samples per epoch: " << trainingSamples << "\n\n";

        const auto trainingStart = std::chrono::steady_clock::now();
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(sampleIndices.begin(), sampleIndices.end(), shuffleGenerator);
            const auto epochStart = std::chrono::steady_clock::now();

            for (std::size_t completed = 0; completed < trainingSamples; ++completed) {
                const std::size_t sampleIndex = sampleIndices[completed];
                network.train(trainingData.inputAt(sampleIndex), trainingData.targetAt(sampleIndex));

                if ((completed + 1) % 1000 == 0 || completed + 1 == trainingSamples) {
                    std::cout << "Epoch " << epoch + 1 << '/' << epochs << ": " << completed + 1 << '/'
                              << trainingSamples << " samples\n";
                }
            }

            const double elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epochStart).count();
            std::cout << "Epoch " << epoch + 1 << " completed in " << elapsedSeconds << " s\n";
        }

        const double totalTrainingSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - trainingStart).count();
        std::cout << "Total training time: " << totalTrainingSeconds << " s\n";

        std::size_t correctPredictions = 0;
        for (std::size_t sampleIndex = 0; sampleIndex < testData.size(); ++sampleIndex) {
            const Matrix prediction = network.forward(testData.inputAt(sampleIndex));
            const Matrix target = testData.targetAt(sampleIndex);

            if (indexOfLargestValue(prediction) == indexOfLargestValue(target)) {
                ++correctPredictions;
            }
        }

        const double accuracy = 100.0 * static_cast<double>(correctPredictions) / testData.size();
        std::cout << "Test accuracy: " << correctPredictions << '/' << testData.size() << " (" << accuracy << "%)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Training failed: " << error.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }
}
