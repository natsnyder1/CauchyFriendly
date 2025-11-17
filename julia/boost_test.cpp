#include <iostream>
#include <boost/multiprecision/cpp_int.hpp>

int main() {
    using boost::multiprecision::cpp_int;

    cpp_int x = 1;
    for (int i = 0; i < 100; ++i) x *= 2;

    std::cout << "2^100 = " << x << "\n";
    return 0;
}

// run sudo apt install libboost-all-dev
