#include "neural_network.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;
int main() {
  double input_array[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double target_array[4][1] = {{0.0}, {1.0}, {1.0}, {0.0}};

  vector<vector<double>> inputs;
  for (int i = 0; i < 4; i++) {
    vector<double> input;
    for (int j = 0; j < 2; j++)
      input.push_back(input_array[i][j]);
    inputs.push_back(input);
    input.clear();
  }

  vector<vector<double>> targets;
  for (int i = 0; i < 4; i++) {
    vector<double> target;
    for (int j = 0; j < 1; j++)
      target.push_back(target_array[i][j]);
    targets.push_back(target);
    target.clear();
  }

  vector<int> layer_sizes;
  layer_sizes.push_back(2);
  layer_sizes.push_back(2);
  layer_sizes.push_back(1);

  double bias = 1.0;
  double learning_rate = 5.0;

  Neural_Network *nn = new Neural_Network(layer_sizes, bias, learning_rate);

  // Training
  for (int i = 0; i <= 10000; i++) {
    double average_error = 0.00;
    for (int j = 0; j < inputs.size(); j++) {
      nn->set_input(inputs[j]);
      nn->set_target(targets[j]);
      nn->feed_forward();
      nn->calculate_errors();
      nn->back_propagation();
      average_error += nn->error;
    }
    average_error = average_error / inputs.size();
    if (i % 1000 == 0)
      cout << "Epoch " << i << " Error: " << average_error << endl;
  }

  // Prediction
  for (int j = 0; j < inputs.size(); j++) {
    nn->set_input(inputs[j]);
    nn->feed_forward();
    cout << inputs[j][0] << " " << inputs[j][1] << " | " << nn->layers.back()
         << endl;
  }
}
