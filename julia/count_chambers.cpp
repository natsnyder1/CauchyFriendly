#include "count_chambers.hpp"
#include <cassert>
#include <cmath>

// tolerance for float comparisons
static constexpr Scalar COL_TOL = 1e-9;

/*************************************************************************
******************************* Helper functions *************************
*************************************************************************/

// Helper to check if hyperplane is zero/degenerate which can occur after intersections
static bool is_zero_column(const Arrangement& A, std::size_t j) {
    for (std::size_t r = 0; r < A.dim; ++r) {
        if (std::fabs(A(r, j)) > COL_TOL) {
            return false;
        }
    }
    return true;
}

// Helper to delete a hyperplane/column j from arrangement A
static Arrangement delete_column(const Arrangement& A, std::size_t j) {
    assert(j < A.num_hp);
    Arrangement B(A.dim, A.num_hp - 1);

    for (std::size_t r = 0; r < A.dim; ++r) {
        std::size_t cB = 0;
        for (std::size_t c = 0; c < A.num_hp; ++c) {
            if (c == j) continue;
            B(r, cB) = A(r, c);
            ++cB;
        }
    }
    return B;
}

// Helper to check if two columns c1 and c2 are proportional
static bool are_proportional(const Arrangement& A,
                             std::size_t c1,
                             std::size_t c2)
{
    assert(c1 < A.num_hp && c2 < A.num_hp);
    const std::size_t d = A.dim;

    // Find a row where at least one column is nonzero
    std::size_t r0 = d;
    for (std::size_t r = 0; r < d; ++r) {
        Scalar v1 = A(r, c1);
        Scalar v2 = A(r, c2);
        if (std::fabs(v1) > COL_TOL || std::fabs(v2) > COL_TOL) {
            r0 = r;
            break;
        }
    }

    // If r0 == d, both columns are all ~0 (we shouldn't pass zeros here
    // if we filter them out first, but treat them as proportional anyway)
    if (r0 == d) {
        return true;
    }

    Scalar v1 = A(r0, c1);
    Scalar v2 = A(r0, c2);

    // One zero, one nonzero -> not proportional
    if (std::fabs(v1) <= COL_TOL && std::fabs(v2) > COL_TOL) return false;
    if (std::fabs(v2) <= COL_TOL && std::fabs(v1) > COL_TOL) return false;

    // Both nonzero, define lambda = v2 / v1
    Scalar lambda = v2 / v1;

    for (std::size_t r = 0; r < d; ++r) {
        Scalar w1 = A(r, c1);
        Scalar w2 = A(r, c2);
        if (std::fabs(w2 - lambda * w1) > COL_TOL) {
            return false;
        }
    }
    return true;
}

// Helper to build an arrangement with duplicate and zero columns removed
static Arrangement deduplicate_columns(const Arrangement& A) {
    if (A.num_hp == 0) {
        return A;
    }

    std::vector<std::size_t> keep_cols;
    keep_cols.reserve(A.num_hp);

    for (std::size_t c = 0; c < A.num_hp; ++c) {
        // Drop completely zero columns (not hyperplanes)
        if (is_zero_column(A, c)) {
            continue;
        }

        bool is_duplicate = false;
        for (std::size_t kc : keep_cols) {
            if (are_proportional(A, kc, c)) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            keep_cols.push_back(c);
        }
    }

    Arrangement B(A.dim, keep_cols.size());
    for (std::size_t r = 0; r < A.dim; ++r) {
        for (std::size_t j = 0; j < keep_cols.size(); ++j) {
            B(r, j) = A(r, keep_cols[j]);
        }
    }

    return B;
}

/*************************************************************************
******************************* Recursive Functions **********************
*************************************************************************/

// Restrict arrangement A to hyperplane j 
static Arrangement restrict_to_hyperplane(const Arrangement& A, std::size_t j) {
    assert(j < A.num_hp);
    std::size_t d = A.dim;
    std::size_t n = A.num_hp;

    // Grab the normal of the hyperplane we are restricting to, the jth hp/col of A
    std::vector<Scalar> nj(d);
    for (std::size_t r = 0; r < d; ++r) {
        nj[r] = A(r, j);
    }

    // Find a pivot coordinate p where nj[p] != 0
    std::size_t p = d;
    for (std::size_t r = 0; r < d; ++r) {
        if (std::fabs(nj[r]) > COL_TOL) {
            p = r;
            break;
        }
    }

    assert(p != d);

    Arrangement B(d - 1, n - 1);
    Scalar np = nj[p];

    // For each hyperplane k != j:
    std::size_t colB = 0;
    for (std::size_t k = 0; k < n; ++k) {
        if (k == j) continue;

        // New normal lives in R^(d-1) (coordinates except p)
        std::size_t rowB = 0;
        for (std::size_t r = 0; r < d; ++r) {
            if (r == p) continue;

            Scalar akr = A(r, k);
            Scalar akp = A(p, k);

            // After eliminating x_p using n_j^T x = 0:
            // a'_k[r] = a_k[r] - a_k[p] * (n_j[r] / n_j[p])
            Scalar coeff = akr - akp * (nj[r] / np);

            B(rowB, colB) = coeff;
            ++rowB;
        }

        ++colB;
    }

    // Deduplicate proportional columns in the restricted arrangement
    return deduplicate_columns(B);
}

// Recursive deletion–restriction R(A) = R(A\H) + R(A^H)
static std::int64_t count_chambers_rec(const Arrangement& A_raw) {
    // Always work on a valid arrangement
    Arrangement A = deduplicate_columns(A_raw);

    // Base cases:
    // no hyperplanes means 1 chamber
    if (A.num_hp == 0) return 1;
    // Base: 0-dimensional space means point means 1 chamber
    if (A.dim == 0) return 1;

    // Choose hyperplane Hj to start recursion (we start with last one)
    std::size_t j = A.num_hp - 1;

    // Delete branch: A\Hj
    Arrangement A_del = delete_column(A, j);
    std::int64_t r_del = count_chambers_rec(A_del);

    // Restrict branch: A^Hj
    Arrangement A_res = restrict_to_hyperplane(A, j);
    std::int64_t r_res = count_chambers_rec(A_res);

    return r_del + r_res;
}

std::int64_t count_chambers(const Arrangement& A) {
    return count_chambers_rec(A);
}


/*************************************************************************
**************************** Main for Testing only ***********************
*************************************************************************/
// int main() {

//     // Test 1: Axes in R^2 => 4 chambers
//     {
//         Arrangement A(2, 2);
//         A(0,0) = 1; A(1,0) = 0;   // x = 0
//         A(0,1) = 0; A(1,1) = 1;   // y = 0

//         std::int64_t chambers = count_chambers(A);
//         std::cout << "Test 1 (axes): expected 4, got " << chambers << "\n";
//     }

//     // Test 2: 3 lines in general position => 6 chambers
//     {
//         Arrangement A(2, 3);
//         A(0,0) = 1; A(1,0) = 0;    // x = 0
//         A(0,1) = 0; A(1,1) = 1;    // y = 0
//         A(0,2) = -1; A(1,2) = 1;   // y = x

//         std::int64_t chambers = count_chambers(A);
//         std::cout << "Test 2 (3 lines): expected 6, got " << chambers << "\n";
//     }

//     // Test 3: From Julia
//     {
//         Arrangement A(2, 4);
//         A(0,0) = -1; A(1,0) = 1; 
//         A(0,1) = 1;  A(1,1) = 0;
//         A(0,2) = 1;  A(1,2) = 1;
//         A(0,3) = 0;  A(1,3) = 1;

//         std::int64_t chambers = count_chambers(A);
//         std::cout << "Test 3 (Julia): expected 8, got " << chambers << "\n";

//         A.print();
//     }

//     return 0;
// }

/*

g++ -std=c++20 count_chambers.cpp -o chambers_test
./chambers_test

*/