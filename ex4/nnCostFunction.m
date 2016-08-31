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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% ====Feedforward and Cost Function====
% --for-loop version--
%{
for i = 1:m
    A1 = X(i,:)';
    % --forward propagation for prediction#begin--
    % A1 = X';
    A1_intercept = [ones(1, size(A1, 2)); A1];

    Z2 = Theta1 * A1_intercept;
    A2 = sigmoid(Z2);
    A2_intercept = [ones(1, size(A2, 2)); A2];

    Z3 = Theta2 * A2_intercept;
    A3 = sigmoid(Z3);

    [M, I] = max(A3);
    predictions = I(:);
    % --forward propagation for prediction#end--
    h_theta = A3;
    yi = (1:num_labels==y(i));
    J = J + ((-yi*log(h_theta)) - (1-yi)*log(1-h_theta)); 
end
J = J / m;
%}

% --vectorized version--
% --forward propagation for prediction#begin--
A1 = X';                                        % 400 * 5000
A1_intercept = [ones(1, size(A1, 2)); A1];      % 401 * 5000

Z2 = Theta1 * A1_intercept;                     % 25 * 5000
A2 = sigmoid(Z2);                               % 25 * 5000
A2_intercept = [ones(1, size(A2, 2)); A2];      % 26 * 5000

Z3 = Theta2 * A2_intercept;                     % 10 * 5000
A3 = sigmoid(Z3);                               % 10 * 5000

[M, I] = max(A3);
predictions = I(:);                             % 5000 * 1
% --forward propagation for prediction#end--

% --compute cost function J without regularization term--
h_Theta_vec = A3(:);    % a 50000 by 1 column vector, stored hypotheses for each example.
y_vec = [];             % a 1 by 50000 row vector, stored labels vector for each example.
for i = 1:m
    y_vec = [y_vec 1:num_labels==y(i)];
end
J = (-y_vec*log(h_Theta_vec) - (1-y_vec)*log(1-h_Theta_vec)) / m;

%% ====Regularized Cost Function====
% --compute the regularziation term of cost function J--
Theta1_ni = Theta1(:, 2:end);   % Theta1 with no intercept terms.
Theta2_ni = Theta2(:, 2:end);   % Theta2 with no intercept terms.
J_rgl_term = lambda/(2*m) * ... % the regulariztion term of cost function J
    (Theta1_ni(:)'*Theta1_ni(:) + Theta2_ni(:)'*Theta2_ni(:));
J = J + J_rgl_term;

%% ====Neural Network Gradient (Backpropagation)====
% --for-loop version--
% --compute the gradient of each Theta using backward propagation--
%{
for i = 1:m
    % --forward propagation--
    xi = X(i, :);
    a1 = xi(:);                     % 400 * 1
    a1_intercept = [1; a1];         % 401 * 1
    
    z2 = Theta1 * a1_intercept;     % 25 * 1
    a2 = sigmoid(z2);               % 25 * 1
    a2_intercept = [1; a2];         % 26 * 1
    
    z3 = Theta2 * a2_intercept;     % 10 * 1
    a3 = sigmoid(z3);               % 10 * 1
    
    % --backward propagation--
    y_logical_vec = 1:num_labels == y(i);   % the logical vector form of label y.
    delta3 = a3 - y_logical_vec(:);                                     % 10 * 1
    Theta2_grad = Theta2_grad + delta3*a2_intercept';                   % 10 * 26
    
    delta2_intercept = (Theta2'*delta3) .* sigmoidGradient([1; z2]);    % 26 * 1
    delta2 = delta2_intercept(2:end);                                   % 25 * 1
    Theta1_grad = Theta1_grad + delta2(:)*a1_intercept';                % 25 * 401
end

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;
%}

% --vectorized version--
% --compute the gradient of each Theta using backward propagation--

Y = reshape(y_vec(:), size(A3));                    % 10 * 5000
Delta3 = A3 - Y;                                    % 10 * 5000
Theta2_grad = Theta2_grad + Delta3*A2_intercept';	% 10 * 26

Delta2_intercept = Theta2'*Delta3 .* ...
    sigmoidGradient([ones(1, size(Z2, 2)); Z2]);	% 26 * 5000
Delta2 = Delta2_intercept(2:end, :);                % 25 * 5000
Theta1_grad = Theta1_grad + Delta2*A1_intercept';   % 25 * 401

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

%% ====Regularized Gradient====
Theta1_regularization = lambda/m*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_regularization = lambda/m*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = Theta1_grad + Theta1_regularization;
Theta2_grad = Theta2_grad + Theta2_regularization;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
