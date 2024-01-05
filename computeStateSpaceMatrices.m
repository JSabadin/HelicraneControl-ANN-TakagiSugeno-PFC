function [Am, Bm, Cm, Rm] = computeStateSpaceMatrices(y_current, y_prev, Am_i, Bm_i, Cm_i, Rm_i, cluster_n,sigmas,centers)
    Am = zeros(2,2);
    Bm = zeros(2,1);
    Cm = zeros(1,2);
    Rm = zeros(2,1);
    
    MF_values = zeros(cluster_n, 1);
    for i = 1:cluster_n
        %                     [y(k-1) ,  y(k-2)]
        MF_i = ((exp(-0.5 * (([y_current y_prev] - centers(i,:)) ./ sigmas(i,:)).^2)));
        MF_values(i) = prod(MF_i); % (AND) operation
    end
    
    MF_total = sum(MF_values);

    % Ensure MF_total is not too close to zero to prevent division errors
    if abs(MF_total) < 1e-12
        error('Total membership function value is too close to zero. Adjust your membership functions or input values.');
    end

    Beta_values = MF_values / MF_total;
    
    for i = 1:cluster_n
        beta = Beta_values(i); % extract current beta value
        
        % Adjust multiplication order for better precision
        Am = Am + beta * Am_i(:,:,i);
        Bm = Bm + beta * Bm_i(:,:,i);
        Cm = Cm + beta * Cm_i(:,:,i);
        Rm = Rm + beta * Rm_i(:,:,i);
    end
end
