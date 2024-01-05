function [U, center, sigmas] = fcm(data, cluster_n, expo)
% data: matrix where each row is a sample
% cluster_n: number of clusters
% expo: exponent for the matrix U

% Initialize parameters
data_n = size(data, 1);
features_n = size(data, 2);
U = initfcm(cluster_n, data_n);
U_old = zeros(size(U));
max_iter = 100;
min_impro = 1e-5;
objFcn = zeros(max_iter, 1);

% Initialize covariance matrices and sigmas
covariance_matrices = cell(1, cluster_n);
sigmas = zeros(cluster_n, features_n);

% Main loop
for i = 1:max_iter
    center = U.^expo * data ./ (sum(U.^expo, 2) * ones(1, features_n));
    dist = distfcm(center, data);
    objFcn(i) = sum(sum((dist.^2) .* U));
    
    % Check termination condition
    if i > 1
        if abs(objFcn(i) - objFcn(i-1)) < min_impro
            break;
        end
    end
    
    % Update U
    tmp = dist.^(-2/(expo-1));
    U = tmp ./ (ones(cluster_n, 1) * sum(tmp));
    
    % If U didn't change significantly, break
    if max(max(abs(U - U_old))) < min_impro
        break;
    else
        U_old = U;
    end
end

% Calculate covariance matrices and sigmas
for j = 1:cluster_n
    % Calculate covariance matrix for each cluster
    u = U(j, :).^expo;
    covariance_matrices{j} = ((u' .* (data - center(j, :)))' * (data - center(j, :))) / sum(u);
    % Calculate sigma (standard deviation) for each cluster and each dimension
    sigmas(j, :) = sqrt(diag(covariance_matrices{j}))';
end

function d = distfcm(center, data)
% Euclidean distance
d = zeros(size(center, 1), size(data, 1));
for k = 1:size(center, 1)
    d(k, :) = sqrt(sum(((data-ones(size(data, 1), 1)*center(k, :)).^2), 2));
end

function U = initfcm(cluster_n, data_n)
% Generate initial fuzzy partition matrix
U = rand(cluster_n, data_n);
col_sum = sum(U);
U = U./col_sum(ones(cluster_n, 1), :);
