%% Settings
filename ='darcy_238.mat';   % Name of the dataset file
s = 238;                % Number of grid points on [0,1]^2 
num_samples = 1200;     % Number of samples

% Parameters of covariance C = tau^(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
alpha = 2;
tau = 3;

%Forcing function, f(x) = 1 
f = ones(s,s);

%% Generation
inputs = zeros(s, s, num_samples);
outputs = zeros(s, s, num_samples);

ppm = ParforProgressbar(num_samples);
parfor samplenum = 1:num_samples
    %disp([num2str(samplenum) "/" num2str(num_samples)])

    %Generate random coefficients from N(0,C)
    norm_a = GRF(alpha, tau, s);

    % Threshhold the coefficients
    thresh_a = zeros(s,s);
    thresh_a(norm_a >= 0) = 12;
    thresh_a(norm_a < 0) = 4;
    
    %Solve PDE: - div(a(x)*grad(p(x))) = f(x)
    p = solve_gwf(thresh_a,f);

    % Store the results
    inputs(:,:,samplenum) = thresh_a;
    outputs(:,:,samplenum) = p;

    % Increment progress bar
    ppm.increment();
end

%% Saving
save(filename, "outputs", "inputs");