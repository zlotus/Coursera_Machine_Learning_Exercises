function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

[m, n] = size(X);     % number of examples & number of features      
Idx_compare = zeros(m, K);

for i = 1:K
    mu_i = centroids(i, :);
    Mu_i = ones(m, 1) * mu_i;
    Idx_compare(:, i) = ((X-Mu_i).^2) * ones(n, 1);
end

[M, I] = min(Idx_compare, [], 2);
idx = I;

% =============================================================

end

