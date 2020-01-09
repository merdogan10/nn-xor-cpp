#include "neural_network.hpp"

Neural_Network::Neural_Network(vector<int> layer_sizes, double bias,
                               double learning_rate) {
  srand(time(NULL));
  this->layer_sizes = layer_sizes;
  this->bias = bias;
  this->learning_rate = learning_rate;

  // Init Layers
  for (int i = 0; i < layer_sizes.size(); i++) {
    layers.push_back(MatrixXd::Zero(layer_sizes[i], 1));
    gradients.push_back(MatrixXd::Zero(layer_sizes[i], 1));
  }

  // Init weights
  for (int i = 0; i < layer_sizes.size() - 1; i++) {
    weights.push_back(MatrixXd::Random(layer_sizes[i + 1], layer_sizes[i] + 1));
    delta_weights.push_back(
        MatrixXd::Zero(layer_sizes[i + 1], layer_sizes[i] + 1));
  }

  // Init errors
  errors = MatrixXd::Zero(layer_sizes.back(), 1);
  derived_errors = MatrixXd::Zero(layer_sizes.back(), 1);
}

void Neural_Network::set_input(vector<double> input) {
  MatrixXd input_layer(input.size(), 1);
  for (int i = 0; i < input.size(); i++) {
    input_layer(i, 0) = input[i];
  }
  layers[0] = input_layer;
}

void Neural_Network::set_target(vector<double> target) {
  this->target = target;
}

MatrixXd Neural_Network::activate(MatrixXd m) {
  MatrixXd n(m.rows(), m.cols());
  // Sigmoid
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      n(i, j) = 1 / (1 + exp(-m(i, j)));
    }
  }
  return n;
}
MatrixXd Neural_Network::derive(MatrixXd m) {
  // Sigmoid derivative
  return (m.array() * (1 - m.array())).matrix();
}

void Neural_Network::feed_forward() {
  for (int i = 0; i < layer_sizes.size() - 1; i++) {
    // Add bias to the end
    layers[i].conservativeResize(layers[i].rows() + 1, layers[i].cols());
    layers[i](layers[i].rows() - 1, 0) = bias;

    layers[i + 1] = (weights[i] * layers[i]).array();
    layers[i + 1] = activate(layers[i + 1]);
  }
}

void Neural_Network::calculate_errors() {
  error = 0.00;
  for (int i = 0; i < layers.back().rows(); i++) {
    double temp_error = layers.back()(i, 0) - target[i];
    error += temp_error * temp_error * 0.5;
    derived_errors(i, 0) = temp_error;
  }
}

void Neural_Network::back_propagation() {
  for (int i = layer_sizes.size() - 1; i > 0; i--) {
    if (i == layer_sizes.size() - 1) {
      // Output Layer
      gradients[i] = derived_errors.cwiseProduct(derive(layers[i]));
      delta_weights[i - 1] = gradients[i] * layers[i - 1].transpose();
    } else {
      // Hidden Layers
      gradients[i] = (weights[i].transpose() * gradients[i + 1])
                         .cwiseProduct(derive(layers[i]));
      // Remove bias from the end
      gradients[i].conservativeResize(gradients[i].rows() - 1,
                                      gradients[i].cols());
      delta_weights[i - 1] = gradients[i] * layers[i - 1].transpose();
    }
  }

  // Update weights
  for (int i = 0; i < weights.size(); i++) {
    weights[i] =
        (weights[i].array() - (learning_rate * delta_weights[i]).array())
            .matrix();
  }
}
