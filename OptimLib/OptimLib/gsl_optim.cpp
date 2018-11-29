#include "pch.h"
#include <iostream>
#include <math.h>
#include <string>
#include <fstream>
#include <cstdlib>
#include <random>
#include <boost/random/non_central_chi_squared_distribution.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <armadillo>
#include "optim.hpp"
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

using namespace std;
using namespace arma;

arma::mat data_raw(1000, 2);
arma::mat R, tau, R0, R1;
int days = 360;

double CIRloglike(const gsl_vector *z, void *params) {
        
    double h = 1. / days; // timestep = 1/360 days;
    double alpha = gsl_vector_get(z, 0);
    double mu = gsl_vector_get(z, 1);
    double sigma = gsl_vector_get(z, 2);
    double c = 2 * alpha / (sigma * sigma * (1 - exp((-alpha * h))));
    double q = 2 * alpha * mu / (sigma * sigma) - 1;
    arma::vec u = c * exp(-alpha * h) * R0;
    arma::vec v = c * R1;
    arma::vec s = 2 * c * R1;
    arma::vec nc = 2 * u;
    arma::vec pdf_matrix_nc2chisq(s.n_rows);
    double df = 2 * q + 2;

    for (int t = 0; t < s.n_rows; ++t) {
        boost::math::non_central_chi_squared_distribution<> d_non_central(df, nc(t));
        pdf_matrix_nc2chisq(t) = pdf(d_non_central, s(t));
    }
    //	for (int i = 0; i < pdf_matrix_nc2chisq.n_rows; i++) {
    //            fi[i] = -log(2 * c * pdf_matrix_nc2chisq[i]); }

    double lnL = accu(-log(2 * c * pdf_matrix_nc2chisq));
    cout << "Loglikelihood value : " << lnL << endl;
    return lnL;
}

int main() {
    try {

        ifstream ip("data.csv");
        if (!ip.is_open())
            std::cout << "ERROR: File Open" << '\n';

        string Tenor;
        string Rate;

        int i = 0;
        double h = 1. / days;
        while (ip.good()) {
            getline(ip, Tenor, ',');
            getline(ip, Rate, '\n');
            data_raw(i, 0) = std::stod(Tenor);
            data_raw(i, 1) = std::stod(Rate);
            ++i;
        }
        ip.close();
        // data_raw.set_size(i, 2);
        //R.set_size(i);
        R1.set_size(i - 1);
        R0.set_size(i - 1);
        //tau.set_size(i);
        for (int t = 0; t < i; ++t) {
            //tau(t) = data_raw(t, 0);
            //R(t) = data_raw(t, 1);
            if (t < i - 1) {
                R1(t) = data_raw(t + 1, 1);
                R0(t) = data_raw(t, 1);
            }
        }
        data_raw.reset();
    } catch (std::exception& e) {
        cout << e.what() << endl;
    }

//   
//    arma::vec x = arma::ones(3, 1) + 1.0; // (2,2)
//    bool success = optim::nm(x, CIRloglike, nullptr);
//    if (success) {
//        std::cout << "nm: LoglikeCIR completed successfully.\n";
//
//    } else {
//        std::cout << "nm: LoglikeCIR completed unsuccessfully." << std::endl;
//    }
//
//    arma::cout << "\n nm: LoglikeCIR solution:\n" << x << arma::endl;
//
//    return 0;

    
  //double par[5] = {1.0, 2.0, 10.0, 20.0, 30.0};

  const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_fminimizer *s = NULL;
  gsl_vector *ss, *x;
  gsl_multimin_function minex_func;

  size_t iter = 0;
  int status;
  double size;

  /* Starting point */
  x = gsl_vector_alloc (3);
  gsl_vector_set (x, 0, 3);
  gsl_vector_set (x, 1, 2);
  gsl_vector_set (x, 2, 1);
  /* Set initial step sizes to 1 */
  ss = gsl_vector_alloc (3);
  gsl_vector_set_all (ss, 0.01);

  /* Initialize method and iterate */
  //minex_func.n = 1;
  minex_func.f = CIRloglike;
  minex_func.n=3;
  //minex_func.params = par;

  s = gsl_multimin_fminimizer_alloc (T, 3);
  gsl_multimin_fminimizer_set (s, &minex_func, x, ss);

  do
    {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);

      if (status)
        break;

      size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, 1e-8);

      if (status == GSL_SUCCESS)
        {
          printf ("converged to minimum at\n");
        }

     
      //cout << iter << "  " << gsl_vector_get (s->x, 0) <<  "   "   << gsl_vector_get (s->x, 1) <<  "   "   gsl_vector_get (s->x, 2) <<  "   "    s->fval, size <<  endl;
    }
  while (status == GSL_CONTINUE && iter < 100);
  
   printf ("Results: \n" );
      cout << gsl_vector_get (s->x, 0) << "\n";
      cout << gsl_vector_get (s->x, 1) << "\n";
      cout << gsl_vector_get (s->x, 2) << "\n";
      cout << "log-likelihood function value: " << s->fval << "\n";
  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free (s);

  return status;
}



