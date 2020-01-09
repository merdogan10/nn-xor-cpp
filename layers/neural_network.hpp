#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;

class Neural_Network {
public:
  Neural_Network(vector<int> layer_sizes, double bias = 1.0,
                 double learning_rate = 0.05);

  vector<int> layer_sizes;
  vector<double> target;

  vector<MatrixXd> layers;
  vector<MatrixXd> gradients;
  vector<MatrixXd> weights;
  vector<MatrixXd> delta_weights;

  MatrixXd errors;
  MatrixXd derived_errors;

  double error = 0.0;
  double bias = 1.0;
  double learning_rate = 0.05;

  void set_input(vector<double> input);
  void set_target(vector<double> target);
  MatrixXd activate(MatrixXd m);
  MatrixXd derive(MatrixXd m);
  void feed_forward();
  void calculate_errors();
  void back_propagation();
};

#endif