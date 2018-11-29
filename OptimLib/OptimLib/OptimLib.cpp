// OptimLib.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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



using namespace std;
using namespace arma;

arma::mat data_raw(1000, 2);
arma::mat R, tau, R0, R1;
int days = 360;

double CIRloglike(const arma::vec& theta, arma::vec* grad_out, void* opt_data) {
    double h = 1. / days; // timestep = 1/360 days;
    double alpha = theta[0];
    double mu = theta[1];
    double sigma = theta[2];
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

    //	real_1d_array theta = "[2,3,1]";
    //	real_1d_array bndl = "[0.001,0.001,0.001]";
    //	real_1d_array bndu = "[100,100,100]";
    //	
    //	double epsx = 0.00000001;
    //	ae_int_t maxits = 0;
    //	minlmstate state;
    //	minlmreport rep;
    //	minlmcreatev(R0.n_rows, theta, 0.00000001, state);
    //	//minlmcreatevj(10, x, state);
    //
    //	minlmsetcond(state, epsx, maxits);
    //
    //	minlmsetbc(state, bndl, bndu);
    //	//		spline1dbuildcubic(T, Y, s);
    //	//		alglib::minlmoptimize(state, NSS);
    //	alglib::minlmoptimize(state, CIRloglike);
    //        
    //	minlmresults(state, theta, rep);
    //	cout << theta.tostring(3).c_str() << endl;

	arma::vec x = { 3,2,1 }; // (2,2)
    bool success = optim::nm(x, CIRloglike, nullptr);
    if (success) {
        std::cout << "nm: LoglikeCIR completed successfully.\n";

    } else {
        std::cout << "nm: LoglikeCIR completed unsuccessfully." << std::endl;
    }

    arma::cout << "\n nm: LoglikeCIR solution:\n" << x << arma::endl;

    return 0;


}

