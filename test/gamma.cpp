#include <iostream>
#include <math.h>
#include "../include/cauchy_estimator.hpp"

void test_procoalign()
{
    const int d = 4;
    const int cmcc = 4;
    double Gamma[d*cmcc] = {
        1,1,-1,-1,
        1,-1,-1,1,
        1,1,-1,-1,
        1,-1,-1,1};
    double beta[cmcc] = {0.1,0.2,0.3,0.4};
    double tmp_Gamma[cmcc*d];
    double tmp_beta[cmcc];

    int tmp_cmcc = precoalign_Gamma_beta(Gamma, beta, cmcc, d, tmp_Gamma, tmp_beta);
    print_mat(Gamma, d, cmcc);
    print_mat(beta, cmcc, 1);
    print_mat(tmp_Gamma, tmp_cmcc, d);
    print_mat(tmp_beta, tmp_cmcc, 1);
}

void test_dce_helper()
{
    const int d = 3;
    const int m_max = 6;
    const int cmcc = 3;
    DiffCellEnumHelper dce_helper;
    dce_helper.init(m_max, d, 2, cmcc); 
    dce_helper.print_tp_info();
    dce_helper.deinit();
}

int main()
{
    //test_procoalign();
    test_dce_helper();
    return 0;
}