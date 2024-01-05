function [U, center, sigmas] = GK(data, cluster_n, expo)
% data: matrix where each row is a sample
% cluster_n: number of clusters
% expo: exponent for the matrix U

% Initialize parameters
data_n = size(data, 1);
features_n = size(data, 2);
U = initGK(cluster_n, data_n);
U_old = zeros(size(U));
max_iter = 100;
min_impro = 1e-9;
d = zeros(cluster_n, data_n);
ro = ones(cluster_n,1);

% Initialize sigmas
sigmas = zeros(cluster_n, features_n);

% Main loop
for i = 1:max_iter
    % 1. calculate the cluster centers
    center = U.^expo * data ./ (sum(U.^expo, 2) * ones(1, features_n));  
    
    for j = 1:cluster_n
        % 2. Calculate fuzzy covariance matrix for each cluster
        u = U(j, :).^expo;
        Cov = ((u' .* (data - center(j, :)))' * (data - center(j, :))) / sum(u);
    
        % Calculate sigma (standard deviation) for each cluster and each dimension
        sigmas(j, :) = sqrt(diag(Cov))';

        [P, D, Q] = svd(Cov);  % Compute SVD
        inv_D = diag(1./diag(D));  % Compute the inverse of D
        Cov_inv = Q*inv_D*P';  % Compute the pseudoinverse

        % Matrix of Inner Product 
        F_j = (ro(j)*det(Cov))^(1/features_n) * Cov_inv;


        % 4. Calculation of Mahalanobis distances
        for k = 1:data_n
            d(j, k) = sqrt((data(k,:) - center(j,:)) * (F_j) * (data(k,:) - center(j,:))');
        end
    end

    ro = 1 ./ (sum(U, 2) * ones(1, features_n));

    % Update U
    tmp = d.^(-2/(expo-1));
    U = tmp ./ (ones(cluster_n, 1) * sum(tmp));

    % If U didn't change significantly, break
    if max(max(abs(U - U_old))) < min_impro
        break;
    else
        U_old = U;
    end

end

function U = initGK(cluster_n, data_n)
% Generate initial fuzzy partition matrix
U = rand(cluster_n, data_n);
col_sum = sum(U);
U = U./col_sum(ones(cluster_n, 1), :);

