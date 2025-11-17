#include "count_chambers.hpp"
#include <cassert>

static Arrangement delete_column(const Arrangement& A, std::size_t j) {
    assert(j < A.n);
    Arrangement B(A.dim, A.n - 1);

    for (std::size_t r = 0; r < A.dim; ++r) {
        std::size_t c2 = 0;
        for (std::size_t c = 0; c < A.n; ++c) {
            if (c == j) continue;
            B(r, c2) = A(r, c);
            ++c2;
        }
    }
    return B;
}

static Arrangement restrict_to_hyperplane(const Arrangement& A, std::size_t j) {
    assert(j < A.n);
    std::size_t d = A.dim;
    std::size_t n = A.n;

    // Copy normals for convenience
    std::vector<Scalar> nj(d);
    for (std::size_t r = 0; r < d; ++r) nj[r] = A(r, j);

    // Find a pivot component in nj
    std::size_t p = d;
    for (std::size_t r = 0; r < d; ++r) {
        if (nj[r] != Scalar(0)) {
            p = r;
            break;
        }
    }
    // If normal is zero (degenerate), just delete it
    if (p == d) {
        return delete_column(A, j);
    }

    // New arrangement lives in dimension d-1, with n-1 hyperplanes
    Arrangement B(d - 1, n - 1);

    Scalar np = nj[p];  // pivot entry

    // For each hyperplane k != j:
    std::size_t colB = 0;
    for (std::size_t k = 0; k < n; ++k) {
        if (k == j) continue;

        // Original normal a_k
        // Equation: sum_r a_k[r] x_r = 0, with constraint sum_r nj[r] x_r = 0
        // Use x_p = -(1/np) * sum_{r != p} nj[r] x_r, substitute into a_k^T x = 0

        // Compute new coefficients in coordinates excluding p
        std::size_t rowB = 0;
        for (std::size_t r = 0; r < d; ++r) {
            if (r == p) continue;

            Scalar akr = A(r, k);

            // contribution from x_r directly
            Scalar coeff = akr;

            // substitute x_p term: a_k[p] * x_p
            Scalar akp = A(p, k);
            // x_p = -(1/np) * sum_{s != p} nj[s] x_s
            coeff -= akp * (nj[r] / np);

            B(rowB, colB) = coeff;
            ++rowB;
        }

        ++colB;
    }

    return B;
}

static int count_chambers_rec(const Arrangement& A) {
    // Base cases
    if (A.n == 0) {
        return 1;  // no hyperplanes in R^d → 1 region
    }
    if (A.dim == 0) {
        return 1;  // 0-dimensional space is just a point
    }

    // Optional: tiny optimization, if all normals are zero → 1 region
    bool allZero = true;
    for (std::size_t r = 0; r < A.dim; ++r)
        for (std::size_t c = 0; c < A.n; ++c)
            if (A(r, c) != Scalar(0))
                allZero = false;
    if (allZero) return 1;

    // Choose a hyperplane index to branch on; simplest: last column
    std::size_t j = A.n - 1;

    // Delete branch
    Arrangement A_del = delete_column(A, j);
    int r_del = count_chambers_rec(A_del);

    // Restrict branch
    Arrangement A_res = restrict_to_hyperplane(A, j);
    int r_res = count_chambers_rec(A_res);

    return r_del + r_res;
}

int count_chambers(const Arrangement& A) {
    return count_chambers_rec(A);
}

