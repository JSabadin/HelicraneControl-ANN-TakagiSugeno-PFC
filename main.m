%% Training signal for Takagi sugeno. Made of Steps spaced logaritmicaly with APRBS added at the end of every step.
clear all; close all; clc;

% Defining constants
ts = 0.01;
total_steps = 9;
start_value = 0.2;
end_value = 1.33;
time_of_step = 5;
samples_per_step = round(time_of_step / ts);
step_size = (end_value - start_value) / total_steps;
T_min = 0.7;
time_of_APRBS = 15;
% Generate logarithmic steps
raw_steps =logspace(-1, 0, total_steps);
raw_steps = flip(abs(raw_steps - 1));
steps = start_value + (end_value - start_value) * (raw_steps - min(raw_steps)) / (max(raw_steps) - min(raw_steps));
% Initializing the signal
signal = [];
% Loop over all steps
for step = 1:total_steps - 1
    % Creating PRBS with random amplitude at each step
    N_train = time_of_APRBS/ts ;
    u = idinput(N_train,'prbs',[0 (T_min/ts)^(-1)],[-1,1]);
    [~,ID] = lastwarn;
    warning('off', ID)
    % Find indices where value changes
    d = diff(u);
    idx = find(d) + 1; %  Index of upper point 
    idx = [1;idx];
    % Assigning random amplitudes for each PRBS segment
    for ii = 1:length(idx) - 1
         rng('shuffle');
         amp = (steps(step)- steps(step + 1)).*rand;
         u(idx(ii):idx(ii+1)-1) = amp*u(idx(ii));  
    end
    % Fixing the signal
    u = abs(u);
    u(idx(end)-1:end) = u(idx(end)-2);
    u = iddata([],u,1);
    u_APRBS = u.InputData';
    u_APRBS = u_APRBS(idx(2):end);
    % Adding the step and the PRBS to the signal
    step_value = steps(step);
    step_signal = repmat(step_value, 1, samples_per_step);
    signal = [signal, step_signal, u_APRBS + steps(step)];
    time_of_APRBS = round(time_of_APRBS*1.2);
    T_min = T_min*1.1;
end
% Adding the final step to the signal
step_value = steps(total_steps);
step_signal = repmat(step_value, 1, samples_per_step);
signal = [signal, step_signal];
% Plotting the signal
t_train = (0:1:length(signal)-1)*ts;
u_train = signal;
plot(t_train,u_train);
xlabel('Time / s');
ylabel('Input voltage / V');
title('Input signal used for training');
% MEASURING HELIOCRANE OUTPUT
% Meausring the output from helicrane device
x = [0 0];
N_samples = length(u_train);
y_train = zeros(1, N_samples);
for i = 1:N_samples
    % Input:
    Fm = u_train(i);
    [fi_, fip_] = helicrane(Fm, x);
    x = [fip_ fi_];
    y_train(i) = fi_; % fi_ is the output of the process we are interested in.
end
figure
plot(t_train,y_train)
xlabel('Time / s');
ylabel('Output angle / degrees');
title('Output measured singal used for training')
%% testing signal 
rng(0);
Time_of_testing = 100;
N_test = Time_of_testing/ts;
T_min = 0.7*5;
u = idinput(N_test,'prbs',[0 (T_min/ts)^(-1)],[-1,1]);
[~,ID] = lastwarn;
warning('off', ID)
a = 0.5; % lower amp limit
b = 1; % upper amp limit
d = diff(u);
idx = find(d) + 1; % changed to find(d)
idx = [1;idx];
for ii = 1:length(idx) - 1
     amp = (b-a).*rand + a;
     u(idx(ii):idx(ii+1)-1) = amp*u(idx(ii));   
end
u = abs(u);
u = iddata([],u,1);
u_test = u.InputData;
u_test = u_test(1:idx(end-1));
u_test(end-1000+1:end) = ones(1000,1)*0.3;
u_test = [u_test;0.5*ones(1000,1);0.7*ones(1000,1);0.9*ones(1000,1);1.2*ones(1000,1);1.3*ones(1000,1)];
t_test = (0:1:length(u_test)-1)*ts;
figure
plot(t_test,u_test);
xlabel('Time / s');
ylabel('Input voltage / V');
title('Input signal used for testing')
% This takes a lot of time. 30 seconds
% MEASURING the output from helicrane device
x = [0 0];
y_test = zeros(1, length(u_test));
for i = 1:length(u_test)
    % Input:
    Fm = u_test(i);
    [fi_, fip_] = helicrane(Fm, x);
    x = [fip_ fi_];
    y_test(i) = fi_; % fi_ is the output of the process we are interested in.
end

%% Static curve

%začetni pogoj:
x = [0 0];
ts = 0.01;%tega se ne da spreminjati!
N = 1000;
u = ones(1,N);
inputs_U = zeros(1,20);
outputs_Y = zeros(1,20);
kot = zeros(1,N);
 for j = 0:34
    x = [0 0];
    U_0 = j*0.04;
    u_ =U_0*u;
    for i = 1:N
        %vhod:
        Fm = u_(i);   
        [fi_ fip_] = helicrane(Fm,x);
        x = [fip_ fi_];
        kot(i+1) = fi_; %fi_ je izhod procesa, ki nas zanima.
    end
    inputs_U(j+1) = U_0;
    outputs_Y(j+1) = kot(end);
 end

 figure
 plot(inputs_U,outputs_Y)
 xlabel('input voltage / V')
 ylabel('output voltage / V')


 %% T-S model
 %% 1. part - CLUSTERING using FCM! or GK...
order = 2;
delay = 0;
overlaping_factor = 1;
% Matrix Psi ((length(y_train)-order) x 4): Psi = [-y(k-1) -y(k-2) u(k-1) u(k-2) 1], where 1 is added for constant term
Psi = [construct_psi(u_train, y_train, order, delay),ones((length(u_train) - order),1)];

% u_train and y_train are training data vectors
N_train = length(u_train);

% Initialize the training data
train_data = zeros(N_train-2, 2);

% Prepare the training data for fuzzy rules
for k = 3:N_train
    train_data(k-2, :) = [y_train(k-1) y_train(k-2)];
end


% fuzzy c-mean clustering
cluster_n = 11;
expo = 2;
[U, centers, sigmas] = fcm(train_data, cluster_n, expo);
%% PLOTTING MF
% Assuming you have [U, centers, C, sigmas] from fcm
warning('off', 'all');
overlapping_factor = 1; % Modify this as per your requirement

% Generate a range for plotting
x_range = linspace(min(train_data(:)), max(train_data(:)), 200);

figure;

colors = lines(cluster_n); % Generate a set of distinct colors for plotting

% Plotting membership function for y(k-1)
subplot(1, 2, 1);

for i = 1:cluster_n
    y_val = exp(-0.5 * ((x_range - centers(i,1)) ./ (sigmas(i,1) * overlapping_factor)).^2);
    plot(x_range, y_val, 'LineWidth', 2, 'Color', colors(i,:));
    hold on;
end

xlabel('y(k-1)');
ylabel('\mu(y(k-1))', 'Interpreter', 'tex');
title('Membership Function for y(k-1)', 'Interpreter', 'tex');

grid on;
legend(arrayfun(@(x) sprintf('Cluster %d', x), 1:cluster_n, 'UniformOutput', false));

% Plotting membership function for y(k-2)
subplot(1, 2, 2);

for i = 1:cluster_n
    y_val = exp(-0.5 * ((x_range - centers(i,2)) ./ (sigmas(i,2) * overlapping_factor)).^2);
    plot(x_range, y_val, 'LineWidth', 2, 'Color', colors(i,:));
    hold on;
end

xlabel('y(k-2)');
ylabel('\mu(y(k-2))', 'Interpreter', 'tex');
title('Membership Function for y(k-2)', 'Interpreter', 'tex');

grid on;
legend(arrayfun(@(x) sprintf('Cluster %d', x), 1:cluster_n, 'UniformOutput', false));

% Adjusting figure for better visualization
sgtitle('Membership Functions in the premise of the rule');

%%
% ... [Your existing code up to generating x_range] ...

% Create a meshgrid for 3D plotting
[X, Y] = meshgrid(x_range, x_range);

Z = zeros(size(X));

% Compute tensor product for corresponding clusters only (diagonal in a conceptual sense, not visual)
for i = 1:cluster_n
    % Compute membership values for x_range for cluster i of y(k-1)
    mu_y_k_1 = exp(-0.5 * ((x_range - centers(i,1)) ./ (sigmas(i,1) * overlapping_factor)).^2);
    
    % Compute membership values for x_range for cluster i of y(k-2)
    mu_y_k_2 = exp(-0.5 * ((x_range - centers(i,2)) ./ (sigmas(i,2) * overlapping_factor)).^2);
    
    % Compute tensor product for this cluster combination
    Z_ij = mu_y_k_1' * mu_y_k_2;
    
    Z = Z + Z_ij;
end

% Plot the 3D tensor product
figure;
surf(X, Y, Z);

xlabel('y(k-1)');
ylabel('y(k-2)');
zlabel('\mu(y(k-1), y(k-2))', 'Interpreter', 'tex');
title('3D Tensor Product of Membership Functions', 'Interpreter', 'tex');
colorbar;



%% 2. part - Creating FUZZY rules and doing the Weighted least squares
% Fuzzy rules
% Initialize a cell array to hold the membership functions
MF = cell(cluster_n, 1);

% For each cluster we get one rule of form:
% IF y(k-1) is MF1 AND y(k-2) is MF2 THEN y(k) = - a1*y(k-1) + - a2 *y(k-2) + b1*u(k-1) + b2*u(k-2) + C
for i = 1:cluster_n
    MF_temp = exp(-0.5 * ((train_data - centers(i,:)) ./ sigmas(i,:)*overlaping_factor).^2);
    MF{i} = prod(MF_temp'); % AND term in fuzzy rules
end

MF_sumed = sum(cell2mat(MF));

% Weighted Least Squares
for i = 1:cluster_n
    % Theta consists of [a1,a2,b1,b2,C]
    theta{i} = ((Psi'.* (MF{i} ./ MF_sumed))*Psi) \ ((Psi'.* (MF{i} ./ MF_sumed))*y_train(order + delay + 1 : end)'); % More stable with \
end

%% Simulating response to the TRAINING signal
N_sim = length(u_train); % total number of simulation steps
y_simulated = zeros(N_sim, 1);

for k = order+1:N_sim
    firing_strengths = zeros(1, cluster_n);
    rule_outputs = zeros(1, cluster_n);
    
    current_data = [y_simulated(k-1), y_simulated(k-2)]; % current input data based on past y values
    
    for i = 1:cluster_n
        % Compute membership values for current rule
        MF_temp = exp(-0.5 * ((current_data - centers(i,:)) ./ (sigmas(i,:)*overlaping_factor)).^2);
        
        % Compute firing strength for current rule (AND operation)
        firing_strengths(i) = prod(MF_temp);
        
        % Construct Psi for current time step
        Psi_sim = [-y_simulated(k-1), -y_simulated(k-2), u_train(k-1), u_train(k-2), 1];
        
        % Compute output based on linear consequent parameters for current rule
        rule_outputs(i) = dot(theta{i}, Psi_sim);
    end
    
    % Normalize firing strengths
    normalized_fs = firing_strengths / sum(firing_strengths);
    
    % Compute weighted average output for current time step
    y_simulated(k) = sum(normalized_fs .* rule_outputs);
end





%% Plotting TRAINING results
t = 0:ts:(length(y_train) - 1)*ts;

figure
plot(t, y_train)
hold on
plot(t,y_simulated)
hold on
e = y_train' - y_simulated;
plot(t,e)

legend('Actual Output', 'Predicted Output', "Error");
legend('Location', 'northwest')
xlabel('Time / s');
ylabel('Output angle / degrees');
title('Comparison of Actual and Predicted Output Signals');

fprintf("\nMean of absolute error for TS on training is %f",mean(abs(e)));



%% Simulating response to the TESTING signal
N_sim = length(u_test); % total number of simulation steps
y_simulated = zeros(N_sim, 1);

for k = order+1:N_sim
    firing_strengths = zeros(1, cluster_n);
    rule_outputs = zeros(1, cluster_n);
    
    current_data = [y_simulated(k-1), y_simulated(k-2)]; % current input data based on past y values
    
    for i = 1:cluster_n
        % Compute membership values for current rule
        MF_temp = exp(-0.5 * ((current_data - centers(i,:)) ./ (sigmas(i,:)*overlaping_factor)).^2);
        
        % Compute firing strength for current rule (AND operation)
        firing_strengths(i) = prod(MF_temp);
        
        % Construct Psi for current time step
        Psi_sim = [-y_simulated(k-1), -y_simulated(k-2), u_test(k-1), u_test(k-2), 1];
        
        % Compute output based on linear consequent parameters for current rule
        rule_outputs(i) = dot(theta{i}, Psi_sim);
    end
    
    % Normalize firing strengths
    normalized_fs = firing_strengths / sum(firing_strengths);
    
    % Compute weighted average output for current time step
    y_simulated(k) = sum(normalized_fs .* rule_outputs);
end


%% Plotting TESTING results
t = 0:ts:(length(y_test) - 1)*ts;

figure
plot(t,y_simulated,'-o')
hold on
plot(t, y_test,"lineWidth",2)
hold on
e = y_test' - y_simulated;
plot(t,e)

legend('Actual Output', 'Predicted Output', "Error");
legend('Location', 'northwest')
xlabel('Time / s');
ylabel('Output angle / degrees');
title('Comparison of Actual and Predicted Output Signals');

fprintf("\nMean of absolute error for TS on training is %f",mean(abs(e)));





%%  Setting up the state space model in SOKO form where we add matrix Rm
% Since the FCM is sensitive to initial setting of U, we save the best
% sigmas and centers
clearvars -except u_test y_test t_test
load('myData.mat'); 
cluster_n = 11;

Am_i = zeros(2,2, cluster_n);  
Bm_i = zeros(2,1, cluster_n); 
Rm_i = zeros(2,1, cluster_n); 
Cm_i = zeros(1,2, cluster_n);  

for i = 1:cluster_n
    % Access the i-th cell of theta and assign the elements to the corresponding variables
    a1 = theta{i}(1);
    a2 = theta{i}(2);
    b1 = theta{i}(3);
    b2 = theta{i}(4);
    C = theta{i}(5);
    
    % Construct the matrix and assign it to the i-th cell of Am_i
    Am_i(:,:,i) = [0, -a2; 1, -a1];
    Bm_i(:,:,i) = [b2;b1];
    Cm_i(:,:,i) = [0 1];
    Rm_i(:,:,i) = [0;C];
end

%% Initialize the reference value signal
u_start = 10;
u_end = 80;
N_parts = 10;
cluster_n = 11;
ts = 0.01;
step_values = [linspace(u_start, u_end, N_parts)];
% step_values = [step_values,flip(step_values(1:end-1))];
time_of_step = 10;
N = time_of_step/ts;
w = [];
% Generate the signal
for i = 1:length(step_values)
    % Append N samples of the current step value to the signal
    w = [w; ones(N, 1) * step_values(i)];
end
t = (0:1:length(w)-1)*ts;

% Reference model state space
s = tf('s');
Gr = 3/(s + 3);  % Continuous system
sys_d = c2d(ss(Gr), ts, 'zoh');  % Convert system to discrete time using Zero-Order Hold
Ar = sys_d.A;
Br = sys_d.B;
Cr = sys_d.C;
% % Iz clanka
% Ar = 0.65;
% Cr = 1;
% Br = 1 - Ar;

%% PFC reference tracking

% Initializing states
x = [0 0];
x_m = [0;
       0];
x_r = 0;

% Initializing input at the beginning
u_k = 0;

% Horizon of prediction
H = 5;

% Initializing signlas 
N = length(w);
p = zeros(N,1);
m = zeros(N,1);
r = zeros(N,1);
u = zeros(N,1);

% Initializing output signals of proces, model and reference
y_p = 0;
y_m = 0;
y_r = 0;
y_del = [0 0];

for k = 1:N
     [Am, Bm, Cm, Rm] = computeStateSpaceMatrices(y_del(1), y_del(2), Am_i, Bm_i, Cm_i, Rm_i, cluster_n,sigmas,centers);
     w_k = w(k);

    % Reference (1. order)
    y_r = Cr * x_r;                % y_r (k)
    r(k) = y_r;

    % Model ouput
    y_m = Cm * x_m;
    y_del = [y_m y_del(1)];

     % Process 
    [fi_ fip_] = helicrane(u_k,x); % y_p (k)
    x = [fip_ fi_];                % x_p (k + 1)
    y_p = fi_;
    p(k) = y_p;

     % Regulator
    G_0 = Cm*(Am^H - eye(2))* pinv(Am - eye(2)) * Bm;
    G = pinv(G_0)*(1 - Ar^H);
    u_next = G * (w_k - y_p) + pinv(G_0)*y_m - pinv(G_0) * Cm * Am^H * x_m - pinv(Bm)*Rm;
    
     % Update the state
    x_m = Am * x_m + Bm * u_k + Rm;                  % x_m (k + 1)
    x_r = Ar * x_r + Br * w_k;                       % x_r (k + 1)

    u_k = u_next;
    u(k) = u_k;
end


e = p - w;
t = (0:1:length(p)-1)*ts;
figure
plot(t,p,'o-')
hold on
plot(t,w)
hold on
plot(t,r,"--")
legend("Proces output","setpoint trajectory","Reference model output",'Location', 'northwest')
ylabel("Output angle / degrees")
xlabel("time / seconds")
title("Reference Tracking")


figure
plot(t,u,"-x")
legend("Input signal",'Location', 'northwest')
ylabel("Input voltage / V")
xlabel("time / seconds")
title("Control Action")

%% PFC Disturbance rejection 20 degrees reference

% Initializing states
x = [0 0];
x_m = [0;
       0];
x_r = 0;

% Initializing input at the beginning
u_k = 0;

% Horizon of prediction
H = 5;

% Reference 
w = 20*ones(1,1500);

% Initializing signlas 
N = length(w);
p = zeros(N,1);
m = zeros(N,1);
r = zeros(N,1);
u = zeros(N,1);

% Initializing output signals of proces, model and reference
y_p = 0;
y_m = 0;
y_r = 0;
y_del = [0 0];

for k = 1:N


    [Am, Bm, Cm, Rm] = computeStateSpaceMatrices(y_del(1), y_del(2), Am_i, Bm_i, Cm_i, Rm_i, cluster_n,sigmas,centers);
    w_k = w(k);

    % Reference (1. order)
    y_r = Cr * x_r;                % y_r (k)
    r(k) = y_r;

    % Model ouput
    y_m = Cm * x_m;
    y_del = [y_m y_del(1)];

     % Process 
    [fi_ fip_] = helicrane(u_k,x); % y_p (k)
    x = [fip_ fi_];                % x_p (k + 1)
    y_p = fi_;
    p(k) = y_p;

     % Regulator
    G_0 = Cm*(Am^H - eye(2))* pinv(Am - eye(2)) * Bm;
    G = pinv(G_0)*(1 - Ar^H);
    u_next = G * (w_k - y_p) + pinv(G_0)*y_m - pinv(G_0) * Cm * Am^H * x_m - pinv(Bm)*Rm;

        % Disturbance
    if k == 901
        u_next = u_next + 0.5;
    end
    
     % Update the state
    x_m = Am * x_m + Bm * u_k + Rm;                  % x_m (k + 1)
    x_r = Ar * x_r + Br * w_k;                       % x_r (k + 1)

    u_k = u_next;
    u(k) = u_k;
end


%%
e = p - w';
N = length(u);
t = (0:1:N-1)*ts;

figure
plot(t(1:500),e(900:1400-1)',"-x")
legend("e[V]",'Location', 'northwest')
ylabel("Output angle / degrees")
xlabel("time / seconds")
title("Disturbance Rejection (Error) at reference of 20 degrees")


figure
plot(t(1:50),u(900:950-1),"-x")
legend("Input signal",'Location', 'northwest')
ylabel("Input voltage / V")
xlabel("time / seconds")
title("Disturbance Rejection (Process input) at reference of 20 degrees")

%% Distrubance rejection 40 degrees reference

% Initializing states
x = [0 0];
x_m = [0;
       0];
x_r = 0;

% Initializing input at the beginning
u_k = 0;

% Horizon of prediction
H = 5;

% Reference 
w = 40*ones(1,1500);

% Initializing signlas 
N = length(w);
p = zeros(N,1);
m = zeros(N,1);
r = zeros(N,1);
u = zeros(N,1);

% Initializing output signals of proces, model and reference
y_p = 0;
y_m = 0;
y_r = 0;
y_del = [0 0];

for k = 1:N


    [Am, Bm, Cm, Rm] = computeStateSpaceMatrices(y_del(1), y_del(2), Am_i, Bm_i, Cm_i, Rm_i, cluster_n,sigmas,centers);
    w_k = w(k);

    % Reference (1. order)
    y_r = Cr * x_r;                % y_r (k)
    r(k) = y_r;

    % Model ouput
    y_m = Cm * x_m;
    y_del = [y_m y_del(1)];

     % Process 
    [fi_ fip_] = helicrane(u_k,x); % y_p (k)
    x = [fip_ fi_];                % x_p (k + 1)
    y_p = fi_;
    p(k) = y_p;

     % Regulator
    G_0 = Cm*(Am^H - eye(2))* pinv(Am - eye(2)) * Bm;
    G = pinv(G_0)*(1 - Ar^H);
    u_next = G * (w_k - y_p) + pinv(G_0)*y_m - pinv(G_0) * Cm * Am^H * x_m - pinv(Bm)*Rm;

        % Disturbance
    if k == 901
        u_next = u_next + 0.5;
    end
    
     % Update the state
    x_m = Am * x_m + Bm * u_k + Rm;                  % x_m (k + 1)
    x_r = Ar * x_r + Br * w_k;                       % x_r (k + 1)

    u_k = u_next;
    u(k) = u_k;
end


%%
e = p - w';
N = length(u);
t = (0:1:N-1)*ts;

figure
plot(t(1:500),e(900:1400-1)',"-x")
legend("e[V]",'Location', 'northwest')
ylabel("Output angle / degrees")
xlabel("time / seconds")
title("Disturbance Rejection (Error) at reference of 40 degrees")


figure
plot(t(1:50),u(900:950-1),"-x")
legend("Input signal",'Location', 'northwest')
ylabel("Input voltage / V")
xlabel("time / seconds")
title("Disturbance Rejection (Process input) at reference of 40 degrees")


%% Disturbance rejcetion 60 degrees reference

% Initializing states
x = [0 0];
x_m = [0;
       0];
x_r = 0;

% Initializing input at the beginning
u_k = 0;

% Horizon of prediction
H = 5;

% Reference 
w = 60*ones(1,1500);

% Initializing signlas 
N = length(w);
p = zeros(N,1);
m = zeros(N,1);
r = zeros(N,1);
u = zeros(N,1);

% Initializing output signals of proces, model and reference
y_p = 0;
y_m = 0;
y_r = 0;
y_del = [0 0];

for k = 1:N


    [Am, Bm, Cm, Rm] = computeStateSpaceMatrices(y_del(1), y_del(2), Am_i, Bm_i, Cm_i, Rm_i, cluster_n,sigmas,centers);
    w_k = w(k);

    % Reference (1. order)
    y_r = Cr * x_r;                % y_r (k)
    r(k) = y_r;

    % Model ouput
    y_m = Cm * x_m;
    y_del = [y_m y_del(1)];

     % Process 
    [fi_ fip_] = helicrane(u_k,x); % y_p (k)
    x = [fip_ fi_];                % x_p (k + 1)
    y_p = fi_;
    p(k) = y_p;

     % Regulator
    G_0 = Cm*(Am^H - eye(2))* pinv(Am - eye(2)) * Bm;
    G = pinv(G_0)*(1 - Ar^H);
    u_next = G * (w_k - y_p) + pinv(G_0)*y_m - pinv(G_0) * Cm * Am^H * x_m - pinv(Bm)*Rm;

        % Disturbance
    if k == 901
        u_next = u_next + 0.5;
    end
    
     % Update the state
    x_m = Am * x_m + Bm * u_k + Rm;                  % x_m (k + 1)
    x_r = Ar * x_r + Br * w_k;                       % x_r (k + 1)

    u_k = u_next;
    u(k) = u_k;
end


%%
e = p - w';
N = length(u);
t = (0:1:N-1)*ts;

figure
plot(t(1:500),e(900:1400-1)',"-x")
legend("e[V]",'Location', 'northwest')
ylabel("Output angle / degrees")
xlabel("time / seconds")
title("Disturbance Rejection (Error) at reference of 60 degrees")


figure
plot(t(1:50),u(900:950-1),"-x")
legend("Input signal",'Location', 'northwest')
ylabel("Input voltage / V")
xlabel("time / seconds")
title("Disturbance Rejection (Process input) at reference of 60 degrees")
%% ANN MODELING
 %% Training signal For ANN Just an APRBS

% Defining APRBS input signal that ranges from 0 to 1.4 volts, with minimum
% step duration of 0.7 seconds which is 70 samples
rng(0);
time_of_experiment = 200;
N_train = time_of_experiment/ts;
% N = 30000;
T_min = 0.7;
u = idinput(N_train,'prbs',[0 (T_min/ts)^(-1)],[-1,1]);
[~,ID] = lastwarn;
warning('off', ID)

a = 0; % lower amp limit
b = 1.33; % upper amp limit
d = diff(u);
idx = find(d) + 1; % changed to find(d)
idx = [1;idx];

for ii = 1:length(idx) - 1
     amp = (b-a).*rand + a;
     u(idx(ii):idx(ii+1)-1) = amp*u(idx(ii));
     
end
u = abs(u);
u = iddata([],u,1);
u_train = u.InputData;
t_train = (0:1:N_train-1)*ts;
figure
plot(t_train,u_train);
xlabel('Time / s');
ylabel('Input voltage / V');
title('Input signal used for training')


% Meausring the output from helicrane device
x = [0 0];
y_train = zeros(1, N_train);
ts = 0.01;
T = 50*ts;

for i = 1:N_train
    % Input:
    Fm = u_train(i);
    [fi_, fip_] = helicrane(Fm, x);
    x = [fip_ fi_];
    y_train(i) = fi_; % fi_ is the output of the process we are interested in.
end

figure
plot(t_train,y_train)
xlabel('Time / s');
ylabel('Output angle / degrees');
title('Output measured singal used for training')

%% Neural network
% TRAINING

order = 2;
delay = 0;
y = y_train(order + delay + 1 : end);
Psi = construct_psi(u_train, y_train, order, delay)';

hiddenLayerSize = 15; 
net = feedforwardnet(hiddenLayerSize);
net = configure(net, Psi, y);

% net.trainParam.epochs = 100; % Set max iterations
net.trainParam.goal = 2e-6;
[net, tr] = train(net, Psi, y);


%% Simulating training response
% Using closed loop SIMULATION for obtaining the output.
% Each iteration 2 delayed outputs and 2 delayed inputs are used to get new
% output value in next time step.
Psi_k = [zeros(1,order)'; u_train(1:order)];
y_s = zeros(length(u_train)-order,1);

% Initialize the waitbar
h = waitbar(0, 'Please wait... Simulating training response...');

for i = 1:length(u_train) - order 
    y_s(i) = -Psi_k(1);

    sim = net(Psi_k);
    Psi_k(2:order) = Psi_k(1:order - 1);
    Psi_k(1) = -sim;
    
    Psi_k(order + 2:end) = Psi_k(order + 1:end-1);
    Psi_k(order + 1) = u_train(i+order);
    
    % Update the waitbar with the current progress
    waitbar(i / (length(u_train) - order), h);
end

% Close the waitbar
close(h);
%%
% Plot the results
figure;
plot(t_train(1:end-order),y_train(order + 1:end));
hold on;
plot(t_train(1:end-order),y_s);
hold on;
plot(t_train(1:end-order),y_train(1: end-order)' - y_s);
legend('Actual Output', 'Predicted Output', "Error");
legend('Location', 'northwest')
xlabel('Time / s');
ylabel('Output angle / degrees');
title('Comparison of Actual and Predicted Output Signals');

fprintf("\nMean of absolute error for ANN on training is %f",mean(abs(y_train(1: end-order)' - y_s)));



% View the network architecture
view(net)

%% Testing Neural Network
% Using closed loop SIMULATION for obtaining the output.
% Each iteration 2 delayed outputs and 2 delayed inputs are used to get new
% output value in next time step.
Psi_k = [zeros(1,order)'; u_test(1:order)];
y_s = zeros(length(u_test)-order,1);

% Initialize the waitbar
h = waitbar(0, 'Please wait... Testing Neural Network...');

for i = 1:length(u_test) - order 
    y_s(i) = -Psi_k(1);

    sim = net(Psi_k);
    Psi_k(2:order) = Psi_k(1:order - 1);
    Psi_k(1) = -sim;
    
    Psi_k(order + 2:end) = Psi_k(order + 1:end-1);
    Psi_k(order + 1) = u_test(i+order);
    
    % Update the waitbar with the current progress
    waitbar(i / (length(u_test) - order), h);
end

% Close the waitbar
close(h);
%%

figure;
plot(t_test(1:end-order),y_test(order + 1 : end), 'o');
hold on;
plot(t_test(1:end-order),y_s, 'LineWIdth',2);
hold on
plot(t_test(1:end-order),y_test(order + 1 : end)' - y_s)
legend('Actual Output', 'Predicted Output',"Error");
legend('Location', 'northwest')
xlabel('Time / s');
ylabel('Ouptut Angle / degrees');
title('Comparison of Actual and Predicted Output Signals');
title('Comparison of Actual and Predicted Output Signals');
fprintf("\nMean of absolute error for ANN on testing is %f",mean(abs(y_test(order + 1 : end)' - y_s)));





