#pragma once

#include <torch/torch.h>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef _MSC_VER
#include <intrin.h>
inline uint32_t bswap32(uint32_t x) {
    return _byteswap_ulong(x);
}
#else
inline uint32_t bswap32(uint32_t x) {
    return __builtin_bswap32(x);
}
#endif

class MNISTDataset : public torch::data::Dataset<MNISTDataset> {
private:
    torch::Tensor images_, targets_;
    size_t dataset_size_;

    torch::Tensor read_images(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to open image file: " + path);

        uint32_t magic_number, num_images, rows, cols;
        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);

        magic_number = bswap32(magic_number);
        num_images = bswap32(num_images);
        rows = bswap32(rows);
        cols = bswap32(cols);

        dataset_size_ = num_images;
        std::vector<uint8_t> buffer(num_images * rows * cols);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        torch::Tensor images = torch::from_blob(buffer.data(), { (int)num_images, 1, (int)rows, (int)cols }, torch::kUInt8).clone();
        return images.to(torch::kFloat32).div_(255); // Normalize to [0,1]
    }

    torch::Tensor read_labels(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to open label file: " + path);

        uint32_t magic_number, num_labels;
        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&num_labels), 4);

        magic_number = bswap32(magic_number);
        num_labels = bswap32(num_labels);

        std::vector<uint8_t> buffer(num_labels);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        torch::Tensor labels = torch::from_blob(buffer.data(), { (int)num_labels }, torch::kUInt8).clone();
        return labels.to(torch::kInt64);
    }

public:
    MNISTDataset(const std::string& image_path, const std::string& label_path) {
        images_ = read_images(image_path);
        targets_ = read_labels(label_path);

        if (images_.size(0) != targets_.size(0)) {
            throw std::runtime_error("Number of images and labels does not match.");
        }
    }

    torch::data::Example<> get(size_t index) override {
        return { images_[index], targets_[index] };
    }

    torch::optional<size_t> size() const override {
        return dataset_size_;
    }
};
