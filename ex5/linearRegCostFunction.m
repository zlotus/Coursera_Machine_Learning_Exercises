function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H_theta = X * theta;
theta_ni = theta(2:end);    % theta without intercept terms.
J = 1/(2*m)*((H_theta-y)'*(H_theta-y)) + ...
    lambda/(2*m)*([0; theta_ni]'*[0; theta_ni]);

grad = 1/m * (X'*(H_theta-y)) + lambda/m * [0; theta_ni];

% =========================================================================

grad = grad(:);

end
