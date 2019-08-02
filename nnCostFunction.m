function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshaping nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setting up some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network.

X = [ones(m,1) X];

    train = zeros(num_labels,m);
    
    for i=1:m
        train(y(i),i) = 1;
    end

    a1 = X';
    a2 = [ones(1,m);sigmoid(Theta1*a1)];
    h = sigmoid(Theta2*a2);
    cost = sum(sum(- (train.*log(h)) - (1-train).*log(1-h)),2);

reg = sum(sum(Theta1(:,2:end).*Theta1(:,2:end)),2) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end)),2);

J = cost/m + lambda/(2*m)*reg;


% Part 2: Implementing the backpropagation algorithm.

del3 = h-train;
del2 = (Theta2'*del3).*sigmoidGradient([ones(1,m);Theta1*a1]); %g'(z)
del2 = del2(2:end,:);
Theta1_grad = del2*a1'/m;
Theta2_grad = del3*a2'/m;


% Part 3: Implementing regularization with the cost function and gradients.


Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));




% Unrolling gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
