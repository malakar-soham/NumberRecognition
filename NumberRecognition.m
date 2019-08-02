%% Initialization
clear ; close all; clc

%% Setting up the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  I am starting by first loading and visualizing the dataset.

% Loading Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data.mat');
m = size(X, 1);

% Randomly selecting 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Pameters ================
%  Implmenting a two layer neural network that classifies digits.

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unrolling parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%% =================== Part 3: Training NN ===================
%  Now I am implementing "fmincg", which
%  is a function which works similarly to "fminunc". These
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);

lambda = 1;

% Creating "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtaining Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 4: Implement Predict =================
%  The Neural Network algorithm learns from all examples and predicts the handwritten numbers

pred = predict(Theta1, Theta2, X);
count = 0;

for i=1:100
    fprintf('...');
end

for i=1:100
    if(count==10)
        fprintf('\n% f',pred(sel(i)));
        count =1;
    else
        fprintf('% f',pred(sel(i)));
        count = count + 1;
    end
end

fprintf('\n');

for i=1:100
    fprintf('...');
end
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred(sel) == y(sel))) * 100);


