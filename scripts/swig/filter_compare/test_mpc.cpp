
#include "mpc_linalg.hpp"

MatArray mpc_make_H(Mat& Q, Mat& Qf, Mat& R, Mat& S)
{
    MatArray H_blocks(3);
    Mat initBlock = R;
    Mat fullBlock = (Q << S) ^ (S.trans() << R);
    Mat finalBlock = Qf;

    H_blocks[0] == initBlock;
    H_blocks[1] == fullBlock;
    H_blocks[2] == finalBlock;
    return H_blocks;
}
MatArray mpc_make_P(Mat& Fx, Mat& Fu, Mat& Ff)
{
    MatArray P_blocks(3);
    Mat initBlock = Fu;
    Mat fullBlock = Fx << Fu;
    Mat finalBlock = Ff;
    P_blocks[0] == initBlock;
    P_blocks[1] == fullBlock;
    P_blocks[2] == finalBlock;
    return P_blocks;
}

MatArray mpc_make_C(Tensor& As, Tensor& Bs)
{
    assert( (As.N == Bs.N) && (As.n == Bs.n) );
    int N = As.N;
    int n = As.n;
    Mat Ident(n,n);
    Ident.eye(n);
    MatArray C_blocks(N);
    C_blocks[0] = -1 * Bs[0] << Ident;
    for(int i = 1; i < N; i++)
        C_blocks[i] = (-1 * As[i] << -1 * Bs[i]) << Ident;
    return C_blocks;
}

Mat mpc_make_b(Mat& A0, Mat& x0, Tensor& cs, int N)
{
    int n = A0.m;
    int b_size = N*n;
    Mat b = Mat(b_size);
    b.zero();
    b.vec_insert(0, A0 & x0 + cs[0] );
    int count = n;
    int i = 1;
    while(count < b_size)
    {
        b.vec_insert(count, cs[i] );
        i += 1;
        count += n;
    }
    return b;
}


Mat mpc_make_h(Mat& x0, Mat& Fx, Mat& Fu, Mat& Ff, Mat& f, Mat& ff, int N)
{
    int n = x0.size();
    int p = Fx.m;
    int m = p-2*n;
    int size_h = (N+1)*p;
    Mat h(size_h);
    h.zero();
    h.vec_insert(0, f - Fx & x0);
    int count = p;
    while(count < size_h-p)
    {
        h.vec_insert(count, f);
        count += p;
    }
    h.vec_insert(count, ff);
    return h;
}

Mat mpc_make_d(Mat& h, MatArray& P_blocks, Mat& z, int N)
{
    int size_h = h.size();
    Mat d(size_h);
    d.zero();
    int p = P_blocks[0].m;
    int m = P_blocks[0].n;
    int n = P_blocks[2].n;
    Mat dsub = d.subvec(0,p);
    Mat hsub = h.subvec(0, p);
    Mat zsub = z.subvec(0, m);
    Mat Psub = P_blocks[0];
    dsub.vec_insert(0,hsub - Psub & zsub);
    int count_d = p;
    int count_z = m;
    Psub = P_blocks[1];
    for(int i = 1; i < N-1; i++)
    {   
        dsub = d.subvec(count_d, p);
        hsub = h.subvec(count_d, p);
        zsub = z.subvec(count_z, n+m);
        dsub.vec_insert(0, hsub - Psub & zsub);
        count_d += p;
        count_z += n + m;
    }
    dsub = d.subvec(N*p, p);
    hsub = h.subvec(N*p, p);
    zsub = z.subvec(N*(n+m)-n, n);
    Psub = P_blocks[0];
    dsub.vec_insert(0, hsub - Psub & zsub);
    //return 1.0 / (h - P @ x)
    return dsub;
}

MatArray mpc_make_phi_inv(MatArray& H_blocks, MatArray& P_blocks, Mat& d, double t, int N)
{
    //return;
    double kappa = 1/t;
    int size_d = d.size();
    MatArray PhiInv_blocks(N+1);
    int p = P_blocks[0].m;
    int m = P_blocks[0].n;
    int n = P_blocks[2].n;

    Mat Psub = P_blocks[0];
    Mat Hsub = H_blocks[0];
    Mat dsub, dPsub, dPSubInner;
    Mat PhiSub, PhiSubInv;

    bool is_H_diag = H_blocks[1].is_diag(1e-12);
    bool is_PTP_diag = P_blocks[1].inner().is_diag(1e-12);
    bool is_diag = is_H_diag * is_PTP_diag;

    // Blocks 1 to N-1
    Hsub = H_blocks[1];
    Psub = P_blocks[1];
    int d_count = p;
    for(int i = 0; i < N+1; i++)
    {   
        if(i==0)
        {
            Hsub = H_blocks[0];
            Psub = P_blocks[0];
        }
        else if(i < N)
        {
            Hsub = H_blocks[1];
            Psub = P_blocks[1];
        }
        else 
        {
            Hsub = H_blocks[2];
            Psub = P_blocks[2];
        }

        dsub = d.extract(d_count,d_count+p);
        dPsub = dsub * Psub;
        if(is_diag)
        {
            PhiSub == Hsub;
            PhiSub.scale_diag(2);
            dPSubInner = dPsub.inner_orth();
            dPSubInner.scale_diag(kappa);
            PhiSub.add_diag( dPSubInner );
            PhiSubInv = PhiSub.inv_diag();
            PhiInv_blocks[i] = PhiSubInv;
        }
        else
        {
            PhiSub = 2 * Hsub + kappa * dPsub.inner();
            PhiSubInv = PhiSub.inv();
            PhiInv_blocks[i] = PhiSubInv;
        }
        d_count += p;
    }
    return PhiInv_blocks;
}

MatArray mpc_make_schur_blocks(MatArray& PhiInv_blocks, Tensor& As, Tensor& Bs)
{
    int N = As.N;
    int n = As[0].n;
    int m = As[0].m;
    MatArray Y_blocks(2*N-1);
    Mat Rtild, Qtild_i, Qtild_im1, Stild;
    Mat A, B, AT, BT;
    Mat Yii, Yij;
    bool is_Stild_zero = PhiInv_blocks[1].extract(n,n+m,0,n).is_zeros(1e-8);

    Rtild = PhiInv_blocks[0].extract(0,m,0,m);
    Qtild_im1 = PhiInv_blocks[1].extract(0,m,0,m); 
    B = Bs[0];
    Y_blocks[0] = B & Rtild & B.trans() + Qtild_im1;
    int count_y = 1;
    for(int i = 1; i < N; i++)
    {
        Rtild = PhiInv_blocks[i].extract(n,n+m,n,n+m);
        Qtild_i = PhiInv_blocks[i+1].extract(0,n,0,n);
        A = As[i];
        B = Bs[i];
        AT = A.trans();
        BT = B.trans();
        Yij = -1 * Qtild_im1 & AT;
        Yii = A & Qtild_im1 & AT + B & Rtild & BT + Qtild_i;
        if(!is_Stild_zero)
        {
            Stild = PhiInv_blocks[i+1].extract(0,n,n,n+m);
            Yij -= Stild & BT;
            Yii += A & Stild & BT + BT & Stild.trans() & AT;
        }
        Y_blocks[count_y++] = Yij;
        Y_blocks[count_y++] = Yii;
        Qtild_im1 = Qtild_i;
    }
    assert(count_y == 2*N-1);
    return Y_blocks;
}

MatArray mpc_make_choleskied_Y_blocks(MatArray& Y_blocks)
{
    int N = Y_blocks.N;
    MatArray L_blocks(N);
    Mat Lii, Lij, Yii, Yij;
    Lii = Y_blocks[0].chol();
    L_blocks[0] = Lii;

    for(int i = 1; i < N; i+=2)
    {
        Yij = Y_blocks[i];
        Yii = Y_blocks[i+1];
        Lij = Lii.solve_triangular(Yij, true).trans();
    }

}

void linear_fast_mpc(Mat& z, Mat& nu, Mat& rt, Mat& x0, Mat& R, Mat& Q, Mat& Qf, Mat& S, Tensor& As, Tensor& Bs, Tensor& cs, Mat& Fx, Mat& Fu, Mat& Ff, Mat& f, Mat& ff)
{
    int n = x0.size();
    int m = R.m;
    int N = As.N;
    double t = 1.0;
    MatArray H_blocks = mpc_make_H(Q,Qf,R,S);
    MatArray P_blocks = mpc_make_P(Fx,Fu,Ff);
    MatArray C_blocks = mpc_make_C(As,Bs);
    Mat h = mpc_make_h(x0, Fx, Fu, Ff, f, ff, N);
    Mat b = mpc_make_b(As[0], x0, cs, N);
    bool iterate_true = true;
    int iterate_count = 0;
    int interate_limit = 20;
    double s = 0.99;

    while(iterate_true)
    {
        Mat d = mpc_make_d(h, P_blocks, z, N);
        MatArray PhiInv_blocks = mpc_make_phi_inv(H_blocks, P_blocks, d, t, N);

    }


    /*
    def linear_fast_mpc(z,nu,rt,x0,R,Q,Qf,S,As,Bs,cs,Fx,Fu,Ff,f,ff):
    n = x0.size 
    m = R.shape[0]
    N = len(As)
    t = 1.0
    # Make Constant Matrices 
    # Make H
    H = mpc_make_H(Q,Qf,R,S,N)
    # Make P 
    P = mpc_make_P(Fx,Fu,Ff,N)
    # Make C 
    C = mpc_make_C(As,Bs,N)
    # Make h 
    h = mpc_make_h(f, ff, Fx, x0, N)
    # Make b 
    b = mpc_make_b(As[0], x0, cs, N)
    # Iterate on residual 
    iterate_true = True
    iterate_count = 0
    s = 0.99 
    while iterate_true:
        # Make d 
        d = mpc_make_d(h, P, z)
        # Make rd 
        rd = mpc_make_rd(rt, z, nu, H, t, P, d, C, S, x0)
        # Make rp 
        rp = mpc_make_rp(C,z,b)
        # Make d
        z_size = N*(n+m)
        nu_size = N*n
        KKT = np.zeros((z_size+nu_size,z_size+nu_size))
        KKT[0:z_size,0:z_size] = H + 1/t * P.T @ np.diag(d**2) @ P
        KKT[0:z_size,z_size:] = C.T
        KKT[z_size:, 0:z_size] = C
        r = np.concatenate((rd,rp))
        delta = - np.linalg.solve(KKT, r)
        dz = delta[0:z_size]
        dnu = delta[z_size:]

        # Backtracking 
        MAX_ITS_LS = 50
        eps = 5e-5
        mu = 5
        delta = 0.1
        alpha = 1.0
        beta = 0.7
        #s = 0.99 * alpha

        # Must keep new point strictly feasible
        effort_counts = 0
        while np.any( mpc_make_d(h,P,z+s*dz) < 0 ):
            s *= beta
            effort_counts += 1
            if effort_counts > MAX_ITS_LS:
                print("Feasibility Check! Cannot Find Feasible New Point!")
                return z, nu
            
        # we then continue to multiply s by beta until we have norm(r_plus)_2 <= (1-alpha*s)*norm(r)_2
        z_new = z + s * dz
        nu_new = nu + s * dnu
        # Make d 
        d = mpc_make_d(h, P, z_new)
        rd_new = mpc_make_rd(rt, z_new, nu_new, H, t, P, d, C, S, x0)
        rp_new = mpc_make_rp(C, z_new, b)
        r_new = np.concatenate((rd_new,rp_new))
        r_old = r 
        is_rnew_smaller = np.linalg.norm(r_new) <=  (1.0 - delta * s) * np.linalg.norm(r_old)
        effort_counts = 0
        while( not is_rnew_smaller):
            if(effort_counts > MAX_ITS_LS):
                print("MAX_ITS_LS Reached! Cannot find a x_plus s.t. norm(r_plus)_2 <= (1-alpha*s)*norm(r)_2...exiting!")
                return z, nu
            effort_counts += 1
            s *= beta 
            z_new = z + s * dz
            nu_new = nu + s * dnu
            # Make d 
            d = mpc_make_d(h, P, z_new)
            rd_new = mpc_make_rd(rt, z_new, nu_new, H, t, P, d, C, S, x0)
            rp_new = mpc_make_rp(C, z_new, b)
            r_new = np.concatenate((rd_new,rp_new))
            is_rnew_smaller = (np.linalg.norm(r_new) <=  (1.0 - delta * s) * np.linalg.norm(r_old))
        # Check residual magnitude
        resid_mag = np.linalg.norm(r_new)
        resid_mag_old = np.linalg.norm(r)
        print("Step {}, Resid Magnitude Old v New: {} vs. {}, backtrack step of s*dz where s={}, barrier t={}".format(iterate_count+1, np.round(resid_mag_old,5), np.round(resid_mag,5),s,t))
        iterate_true = resid_mag > eps
        # Increase barrier parameter, update opt variables
        t *= mu
        z = z_new.copy() 
        nu = nu_new.copy()
        # Check if barrier param is too large 
        iterate_true *= (1/t > eps**2)
        # Increase iteration count
        iterate_count += 1
    return z, nu
    */

}

class FastNonlinMPC
{
    private:
    
    Mat (*dyn_f)(Mat x, Mat u);
    Mat (*dyn_fx)(Mat x, Mat u);
    Mat (*dyn_fu)(Mat x, Mat u);
    int n,m,p;
    Mat Q;
    Mat Qf;
    Mat R;
    Mat S;
    Mat Fx;
    Mat Fu;
    Mat Ff;
    Mat f;
    Mat ff;
    Mat ref_traj;
    Mat x0;
    Mat u0;
    double dt;
    Mat* As;
    Mat* Bs;
    Mat* rt;

    public:
    FastNonlinMPC()
    {

    }



};

void test_linalg()
{
    srand(10);
    Mat A = randn(3,3);
    Mat B = randn(3,3);
    Mat c = randn(3,1);
    printf("A:\n");
    A.print(10, false);
    printf("B:\n");
    B.print(10, false);
    printf("c:\n");
    c.print(10, false);
    Mat C = A & B;
    C = B & C;
    printf("B @ A @ B\n");
    C.print();
    printf("A @ B:\n");
    (A & B).print();
    printf("A @ c:\n");
    (A & c).print();
    printf("A * c:\n");
    (A * c).print();
    printf("A * c.T:\n");
    (A * c.trans()).print();
    printf("A.I @ B:\n");
    (A / B).print();
    printf("A+B:\n");
    (A + B).print();
    printf("A-B:\n");
    (A - B).print();
    printf("A @ B - B.T @ A:\n");
    ((A & B) - (B.trans() & A) ).print();
    printf("A @= A:\n");
    A &= A;
    A.print();
    printf("B += B:\n");
    B += B;
    B.print();
    printf("c & c.T\n");
    (c & c.trans()).print();
    ///*
    Mat D = A * 2.0 + c * B; 
    Mat E = D * 2.0;
    printf("D:\n");
    D.print();
    printf("E:\n");
    E.print();
    E = A & B;
    E.print();
    E = E & D;
    E.print();

    Mat F;
    F == D; // deep copy
    printf("F:\n");
    F.print();
    F *= 2;
        printf("F:\n");
    F.print();
    printf("D:\n");
    D.print();

    // Building operations 
    Mat G = F.copy();
    Mat H = D.copy();
    G <= G;
    H <= H;
    G ^= H;
    printf("[F,F;D,D]:\n");
    G.print();

    // Embed operations
    int nI = 3;
    Mat Ident(nI,nI);
    Ident.eye(nI);
    Mat T = (Ident << Ident) ^ (Ident << Ident);
    
    printf("I Block\n");
    T.print();
    T.embed(0,3,F);
    T.embed(3,0,D);
    T.print();
    (T & T.trans()).inv().print();
    
    const int qm = 2;
    const int qn = 2;
    double _J[qm*qn] = {1.0,2.0,3.0,4.0};
    Mat J(_J,qm,qn);
    J.print();
    
    // Tensor
    int N = 3; 
    int m = 3;
    int n = 3;
    Tensor Tens(N, m, n);
    Tens[0] == F;
    Tens[1] = D;
    Tens[2] = Ident;
    Tens.print();
    F *= 2;
    Tens.print();

    Mat I3 = T(0,3,0,3);
    I3.print();
    
    // F is not orthogonal but will serve the purpose for the diagonal being correct
    F.print();
    F.inner_orth().print();
    F.outer_orth().print();
    
    // Solve Triangular
    const int size_z = 9;
    double _Z[size_z] = {1,0,0,1,2,0,1,2,3};
    Mat Z(_Z, 3, 3);
    const int size_w = 12;
    double _W[size_w] = {1,2,3,4,5,6,7,8,9,10,11,12};
    Mat W(_W, 3, 4);
    Z.print();
    W.print();
    Z.solve_triangular(W, true).print();

    Z.trans().solve_triangular(W, false).print();
    
    Mat foo = ((Z.solve_triangular(C, true).trans() & A) + B).inv();
    foo = foo & A;
    foo &= B + D;
    foo.print();
}

void test_linalg2()
{
    Mat A = randn(5,3);
    Mat B = A;
    Mat C;
    C = A & B.trans();
    Mat D = C;
    Mat F;
    F = B;
    B = C;
    A = C;
    B = D;
    D ^= C;
    D ^= C;
    Mat E = D & D.trans();
    E.print();
}

void test_mpc()
{
    return;
}

int main()
{
    test_linalg();
    test_linalg2();
    //test_mpc();
    printf("sizeof(SharedPtr): %lu\n", sizeof(SharedPtr));
    printf("sizeof(Mat): %lu\n", sizeof(Mat));
    printf("sizeof(Tensor): %lu\n", sizeof(Tensor));
    printf("sizeof(MatArray): %lu\n", sizeof(MatArray));
    return 0;
}

