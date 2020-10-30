
	std::array<std::complex<double>, 13*13> R_reference_values = {
	  0,  0,  0,  0,  +8.552e-01,  +8.552e-01,  -6.794e-02,  -6.794e-02,  +2.723e-18,  +5.243e-18,  +1.876e-16,  -6.118e-16,  0, 
	  +1.000e+00,  0,  0,  0,  +3.315e-01,  +3.315e-01,  -3.426e-01,  -3.426e-01,  +1.372e-17,  +1.560e-17,  -1.221e-15,  +4.183e-15,  0, 
	  0,  +1.000e+00,  0,  0,  +3.315e-01,  +3.315e-01,  -3.426e-01,  -3.426e-01,  +1.372e-17,  +1.560e-17,  -1.221e-15,  +4.183e-15,  0, 
	  0,  0,  0,  0,  -3.927e-39,  -1.426e-38,  +4.544e-38,  -7.160e-37,  +1.763e-23,  -2.057e-23,  +4.344e-05,  -1.464e-04,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  +1.000e+00, 
	  0,  0,  0,  0,  +2.666e-16,  +5.026e-16,  -2.715e-15,  +3.108e-14,  +1.000e+00,  +1.000e+00,  +1.000e+00,  +1.000e+00,  0, 
	  0,  0,  0,  0,  -1.111e-07,  +1.111e-07,  -2.915e-08,  +2.915e-08,  +1.671e-24,  -9.028e-25,  +1.655e-22,  +5.618e-22,  0, 
	  0,  0,  0,  0,  -1.162e-45,  -4.220e-45,  +1.345e-44,  -2.119e-43,  -3.071e-30,  -6.703e-30,  +1.047e-11,  +3.530e-11,  0, 
	  0,  0,  0,  0,  +4.649e-23,  +1.689e-22,  -5.381e-22,  +8.478e-21,  -2.411e-07,  +2.411e-07,  +2.411e-07,  -2.411e-07,  0, 
	  0,  0,  0,  0,  -2.210e-01,  -2.210e-01,  +8.722e-01,  +8.722e-01,  -4.487e-17,  -4.205e-17,  +2.727e-15,  -9.377e-15,  0, 
	  0,  0,  0,  0,  +8.655e-09,  -8.655e-09,  +1.128e-07,  -1.128e-07,  -5.637e-24,  +4.968e-24,  -5.503e-22,  -1.895e-21,  0, 
	  0,  0,  0,  +1.000e+00,  -1.076e-46,  +3.907e-46,  +2.488e-45,  +3.920e-44,  +8.501e-31,  +9.916e-31,  -2.095e-12,  -7.060e-12,  0, 
	  0,  0,  +1.000e+00,  0,  +7.483e-24,  -1.359e-23,  -1.515e-22,  -1.704e-21,  +4.822e-08,  -4.822e-08,  -4.822e-08,  +4.822e-08,  0
	};

	std::array<std::complex<double>, 13*13> chi_reference_values = {
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  +1.000e+00,  0,  0,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  +1.000e+00,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  +1.000e+00,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  +1.000e+00,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
	};

	std::array<std::complex<double>, 13*13> R_inv_reference_values = {
	  -2.920e-01,  +1.000e+00,  0,  +6.119e-12,  0,  +2.585e-18,  +2.812e-09,  +5.028e-20,  -4.521e-12,  +3.700e-01,  +1.758e-09,  0,  0, 
	  -2.920e-01,  0,  +1.000e+00,  +6.119e-12,  0,  +2.585e-18,  +2.812e-09,  +5.028e-20,  -4.521e-12,  +3.700e-01,  +1.758e-09,  0,  0, 
	  +1.845e-23,  0,  0,  +2.821e-18,  0,  -6.617e-24,  +7.260e-17,  +1.357e-11,  +2.000e-01,  +1.550e-22,  +1.128e-15,  0,  +1.000e+00, 
	  -1.711e-45,  0,  0,  0,  0,  +5.654e-32,  +1.364e-39,  +2.000e-01,  +1.213e-24,  -7.726e-45,  +2.031e-38,  +1.000e+00,  0, 
	  +5.967e-01,  0,  0,  -5.384e-12,  0,  +1.011e-18,  -4.593e+06,  +9.704e-06,  -1.800e-12,  +4.648e-02,  -1.187e+06,  0,  0, 
	  +5.967e-01,  0,  0,  -5.551e-12,  0,  -1.724e-18,  +4.593e+06,  -9.704e-06,  -4.980e-12,  +4.648e-02,  +1.187e+06,  0,  0, 
	  +1.512e-01,  0,  0,  -3.819e-11,  0,  +2.620e-17,  +3.524e+05,  +2.354e-04,  -1.018e-10,  +5.851e-01,  +4.524e+06,  0,  0, 
	  +1.512e-01,  0,  0,  -3.766e-11,  0,  +2.344e-17,  -3.524e+05,  -2.354e-04,  +9.340e-11,  +5.851e-01,  -4.524e+06,  0,  0, 
	  +3.826e-16,  0,  0,  +3.415e+03,  0,  +5.000e-01,  -1.139e-11,  -1.416e+10,  -2.074e+06,  +1.340e-15,  -7.991e-09,  0,  0, 
	  -5.130e-15,  0,  0,  -1.151e+04,  0,  +5.000e-01,  +1.084e-08,  -4.774e+10,  +2.074e+06,  -1.797e-14,  +1.606e-07,  0,  0, 
	  +1.347e-48,  0,  0,  +1.151e+04,  0,  +2.502e-19,  +3.710e-28,  +4.774e+10,  +1.271e-12,  +4.619e-48,  +1.855e-27,  0,  0, 
	  -1.604e-50,  0,  0,  -3.415e+03,  0,  +6.421e-20,  +9.521e-29,  +1.416e+10,  -1.638e-13,  -4.276e-50,  +4.760e-28,  0,  0, 
	  0,  0,  0,  0,  +1.000e+00,  0,  0,  0,  0,  0,  0,  0,  0
	};

	std::array<std::complex<double>, 13*13> godunov_reference_values = {
	  +5.000e-01,  0,  0,  +8.929e-14,  0,  +4.457e-19,  -3.952e+06,  -1.640e-05,  -2.693e-13,  +2.620e-17,  -1.323e+06,  0,  0, 
	  +1.460e-01,  0,  0,  -2.941e-12,  0,  -1.780e-18,  -1.643e+06,  -1.835e-05,  +5.823e-12,  -1.850e-01,  -1.943e+06,  0,  0, 
	  +1.460e-01,  0,  0,  -2.941e-12,  0,  -1.780e-18,  -1.643e+06,  -1.835e-05,  +5.823e-12,  -1.850e-01,  -1.943e+06,  0,  0, 
	  +1.127e-38,  0,  0,  +5.000e-01,  0,  -5.863e-25,  +1.991e-32,  -2.074e+06,  -1.258e-17,  +5.003e-38,  -3.424e-34,  0,  0, 
	  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
	  +1.311e-16,  0,  0,  -9.669e-12,  0,  +5.000e-01,  -2.193e-09,  -1.322e-04,  -2.074e+06,  -2.361e-16,  -2.059e-08,  0,  0, 
	  -7.070e-08,  0,  0,  -2.011e-19,  0,  -4.060e-26,  +5.000e-01,  -6.091e-15,  -2.975e-19,  -2.222e-08,  -7.918e-17,  0,  0, 
	  +1.652e-46,  0,  0,  -1.205e-07,  0,  +7.312e-31,  +1.347e-38,  +5.000e-01,  +5.863e-25,  +3.698e-45,  +1.035e-37,  0,  0, 
	  -1.459e-22,  0,  0,  +7.884e-18,  0,  -1.205e-07,  -4.004e-16,  +8.583e-12,  +5.000e-01,  -6.357e-22,  -5.627e-16,  0,  0, 
	  -2.538e-16,  0,  0,  -2.510e-13,  0,  +1.959e-19,  +1.323e+06,  +7.094e-05,  +4.646e-12,  +5.000e-01,  +4.208e+06,  0,  0, 
	  +2.222e-08,  0,  0,  +2.096e-18,  0,  +1.459e-25,  -1.348e-16,  -1.218e-13,  +1.900e-19,  +6.640e-08,  +5.000e-01,  0,  0, 
	  +6.373e-46,  0,  0,  +2.411e-08,  0,  -2.827e-32,  +6.891e-40,  -1.000e-01,  -6.065e-25,  +2.590e-45,  +1.230e-39,  0,  0, 
	  +9.988e-27,  0,  0,  -1.327e-18,  0,  +2.411e-08,  -8.829e-17,  -2.822e-12,  -1.000e-01,  -2.365e-23,  -1.079e-15,  0,  0
	};

	using Matrix13 = typename arma::Mat<std::complex<double>>::template fixed<13, 13>;
	auto test = [](const std::array<std::complex<double>, 13*13>& reference_values, const Matrix13& other_matrix) {
	  const Matrix13 reference_matrix(reference_values.data());
	  const auto difference = arma::norm(reference_matrix.t() - other_matrix);
	  const auto norm = arma::norm(reference_matrix);
	  //we just saved reference with an accuracy of 4 digits.
	  constexpr double tol = 1e-3;
	  return difference < tol*norm;
	};
	const auto R_test = test(R_reference_values, R);
	const auto chi_test = test(chi_reference_values, chi_plus);
	const auto R_inv_test = test(R_inv_reference_values, R_inv);
	const auto godunov_test = test(godunov_reference_values, godunov_plus);
	if (!godunov_test) {
	  std::cout << R_test << chi_test << R_inv_test << godunov_test << std::endl;
	}