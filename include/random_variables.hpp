#ifndef _RANDOM_VARIABLES_HPP_
#define _RANDOM_VARIABLES_HPP_

#include <math.h>
#include <stdlib.h>

#define PI M_PI

// y = L @ x with L lower triangular
void lt_matvecmul(const double* L, const double* x, double* y, const int n)
{
    for(int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for(int j = 0; j <= i; j++)
            sum += L[i*n + j] * x[j];
        y[i] = sum;
    }
}

//return uniform variable on the semi-open interval (0,1] (with low open)
double random_uniform()
{
   // return a uniformly distributed random value
   return ( (double)(rand()) + 1.0 )/( (double)(RAND_MAX) + 1.0 );
}

//return uniform variable on the semi-open interval [0,1) with high open
double random_uniform_hopen()
{
   // return a uniformly distributed random value
   return ( (double)(rand()) )/( (double)(RAND_MAX) + 1.0 );
}

//return uniform variable on the open interval (0,1)
double random_uniform_open()
{
    // return a uniformly distributed random value
    return ( (double)(rand()) + 1.0 )/( (double)(RAND_MAX) + 2.0 );
}

//return uniform variable on the closed interval [0,1]
double random_uniform_closed()
{
    // return a uniformly distributed random value
    return ( (double)rand() )/( (double)(RAND_MAX)  );
}

// mu is the scalar mean value 
// sigma is the scalar standard deviation
double random_normal(double mu, double sigma)
{
   // return a normally distributed random value
   double v1=random_uniform(); // must be between (0,1]
   double v2=random_uniform();
   double unit_randn = cos(2.0*PI*v2)*sqrt(-2.0*log(v1)); //cos(2*3.14*v2)*sqrt(-2.0*log(v1));
   return unit_randn*sigma + mu;
}

// Draw a sample from the exponential pdf f(x) = \lambda * exp(-\lambda * x)
double random_exponential(double lambda)
{
    // Draw a random uniform variable on the open interval (0,1)
    double U = random_uniform_open();
    return -log( U ) / lambda;
}

// Draw a random alpha stable variable 
// the parameters are: 
// 1.) alpha \in (0,2] -- this is the stability param (2=Gaussian, 1 = Cauchy, 0.5 = Levy)
// 2.) beta \in [-1,1] -- this is the skewness param 
// 3.) c \in (0, inf] -- this is the scale param (standard deviation for Gaussian)
// 4.) mu \in [-inf,inf] -- this is the location parameter
// Note: For any value of alpha less than or equal to 2, the variance is undefined 
// This implements the Chambers, Mallows, and Stuck (CMS) method from their seminal paper in 1976
double random_alpha_stable(double alpha, double beta, double c, double mu)
{
    //Generate a random variable on the open interval (-pi/2, pi/2)
    double U = random_uniform_open() * PI  - PI / 2.0;
    //Generate a random exponential variable with mean of 1.0
    double W = random_exponential(1.0);
    double zeta = -beta * tan(PI * alpha / 2.0);
    // X ~ S_\alpha(\beta,1,0)
    // Y = c*X + mu
    double xi, X, Y;
    if(alpha == 1.0)
        xi = PI / 2.0; 
    else 
        xi = 1.0 / alpha * atan(-zeta);
    if(alpha == 1.0)
        X =  1.0 / xi * ( (PI / 2.0 + beta * U) * tan(U) - beta * log((PI/2.0 * W * cos(U)) / (PI/2.0 + beta * U)) );
    else
        X = pow(1.0 + pow(zeta,2), 1.0/(2.0*alpha) ) * sin(alpha*(U+xi)) / pow(cos(U), 1.0/alpha) * pow( cos(U - alpha*(U + xi)) / W, (1.0 - alpha) / alpha );
    // Now scale and locate the random variable depending on if alpha == 1.0 or not 
    if(alpha == 1.0)
        Y = c*X + (2.0/PI)*beta*c*log(c) + mu;
    else
        Y = c*X + mu;
    return Y;
}

// Draw a random symmetric alpha stable variable 
// the parameters are: 
// 1.) alpha \in (0,2] -- this is the stability param (2=Gaussian, 1 = Cauchy, 0.5 = Levy)
// 2.) c \in (0, inf] -- this is the scale param (standard deviation for Gaussian)
// 3.) mu \in [-inf,inf] -- this is the location parameter
// Note: For any value of alpha less than or equal to 2, the variance is undefined 
// This implements the Chambers, Mallows, and Stuck (CMS) method from their seminal paper in 1976
double random_symmetric_alpha_stable(double alpha, double c, double mu)
{
    //Generate a random variable on the open interval (-pi/2, pi/2)
    double U = random_uniform_open() * PI  - PI / 2.0;
    //Generate a random exponential variable with mean of 1.0
    double W = random_exponential(1.0);
    // X ~ S_\alpha(\beta,1,0)
    // Y = c*X + mu
    double xi, X, Y;
    if(alpha == 1.0)
        xi = PI / 2.0; 
    else 
        xi = 0.0;
    if(alpha == 1.0)
        X = tan(U);
    else
        X = sin(alpha*(U+xi)) / pow(cos(U), 1.0/alpha) * pow( cos(U - alpha*(U + xi)) / W, (1.0 - alpha) / alpha );
    // Now scale and locate the random variable depending on if alpha == 1.0 or not 
    Y = c*X + mu;
    return Y;
}

// This function returns a sample from a general multivariate normal distribution, 
// Warning: This function will fail if the cholesky decomposition of the variance is not taken before calling this function,
// x \in R^n - Output: drawn samples with mean mu and variance chol_variance @ chol_variance.T,
// mu \in R^n - The mean vector of the multivariate normal,
// chol_variance \in R^(n x n) - The cholesky decomposition of the variance matrix,
// work \in R^n - Memory allocation for needed intermediate computation.
void multivariate_random_normal(double* x, const double* mu, const double* chol_variance, double* work, const int n)
{
    for(int i = 0; i < n; i++)
        work[i] = random_normal(0, 1);
    lt_matvecmul(chol_variance, work, x, n); // x contains L @ v
    for(int i = 0; i < n; i++)
        x[i] += mu[i]; // x contains L @ v + mu (the drawm M.V sample)
}

// returns a sample from the cauchy pdf with zero median and scaling parameter beta
double random_cauchy(double beta)
{
  return beta * random_normal(0.0, 1.0) / random_normal(0.0, 1.0); //median of cauchy RV returned is zero 
}

// sample_mean \in R^(dx1) -- OUTPUT -- the mean of the given samples
// samples \in R^(num_samples x d) pointer to the num_samples d-dimentional variables (the length of samples is num_samples * d)
// num_samples is the number of samples in the samples vector
// d is the size of the random vector (or variable if d==1)
void mean(double* sample_mean, const double* samples, const int num_samples, const int d)
{
    // Set the sample mean to zero
    for(int i = 0; i < d; i++)
        sample_mean[i] = 0.0;
    
    // Sum together num_samples
    for(int i = 0; i < num_samples; i++)
        for(int j = 0; j < d; j++)
            sample_mean[j] += samples[i*d + j];
    
    // Now average the sample means 
    for(int i = 0; i < d; i++)
        sample_mean[i] /= num_samples;
}

// This function computes the sample covariance over a set of (possibly vector) samples given,
// PSUEDO CODE: sample_covariance = 1 / num_samples * SUM( (samples_i - sample_mean) @ (samples_i - sample_mean).T for i in range(num_samples) ),
// sample_covariance \in R^(d x d) -- OUTPUT -- the covariance of the samples,
// sample_mean \in R^(d x 1) -- must call mean() function first to get this value,
// samples \in R^(num_samples x d) -- the samples, each of dimention d,
// num_samples is the number of samples in the sample list,
// d is the dimention of the random vector,
void covariance(double* sample_covariance, const double* sample_mean, const double* samples, const int num_samples, const int d)
{
    double* work = (double*) malloc(d*sizeof(double));
    double* work2 = (double*)malloc(d * d * sizeof(double));

    for(int i = 0; i < d*d; i++)
    {
        sample_covariance[i] = 0.0;
        work2[i] = 0.0;
    }
    
    // Sum together the individual samples covariance with the mean
    for(int i = 0; i < num_samples; i++)
    {
        // Compute e = x-\bar{x}
        for(int j = 0; j < d; j++)
            work[j] = samples[i*d + j] - sample_mean[j];
        
        // compute P_i = e @ e.T
        for(int j = 0; j < d; j++)
            for(int k = 0; k < d; k++)
                work2[j*d + k] = work[j] * work[k];
        // Sum P_i to sample_covariances
        for(int j = 0; j < d*d; j++)
            sample_covariance[j] += work2[j];
    }

    // Average the sample covariance by the number of samples taken
    for(int i = 0; i < d*d; i++)
        sample_covariance[i] /= num_samples;
	free(work);
	free(work2);
}

// This function computes the covariance of a single (possibly vector) data point using its associated mean x_bar (possibly vector) with formula,
// PSUEDO CODE: cov = (x - x_bar) @ (x - x_bar).T
// x_bar \in R^(d x 1) -- is the mean,
// x \in R^(d x 1) -- is the sample
// d is the size of the vector's x_bar and x
// work \in R^(d x 1) --  Memory allocation for needed intermediate computation.
void covariance(double* cov, const double* x_bar, const double* x, double* work, const int d)
{
    // Compute e = x-\bar{x}
    for(int i = 0; i < d; i++)
        work[i] = x[i] - x_bar[i];
    
    // compute P_i = e @ e.T
    for(int i = 0; i < d; i++)
        for(int j = 0; j < d; j++)
            cov[i*d + j] = work[i] * work[j];
}

#endif // 