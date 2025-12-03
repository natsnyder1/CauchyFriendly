#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <iostream>

using Scalar = double;   

// Central hyperplane arrangement: columns are hyperplane normals
struct Arrangement {
    std::size_t dim;               // dimension d
    std::size_t num_hp;            // number of hyperplanes
    std::vector<Scalar> data;      // row-major storage: data[row * num_hp + col]

    // Empty constructor
    Arrangement() : dim(0), num_hp(0) {}
    // Normal constructor, start with all entries 0
    Arrangement(std::size_t d, std::size_t num)
        : dim(d), num_hp(num), data(d * num, Scalar(0)) {}

    // Access operators
    Scalar& operator()(std::size_t row, std::size_t col) {
        return data[row * num_hp + col];
    }
    const Scalar& operator()(std::size_t row, std::size_t col) const {
        return data[row * num_hp + col];
    }

    // Print out an arrangement using A.print();
    void print(std::ostream& os = std::cout) const {
        os << "Arrangement: dim = " << dim
           << ", num_hp = " << num_hp << '\n';

        for (std::size_t j = 0; j < num_hp; ++j) {
            os << "H_" << j << " normal = [";
            for (std::size_t r = 0; r < dim; ++r) {
                os << (*this)(r, j);
                if (r + 1 < dim) os << ", ";
            }
            os << "]^T\n";
        }
    }
};

// Top-level API:
std::int64_t count_chambers(const Arrangement& A);

// Overload << operator to be able to print out arrangements easily
inline std::ostream& operator<<(std::ostream& os, const Arrangement& A) {
    A.print(os);
    return os;
}