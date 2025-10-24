%% Polywell Fusion Simulation with Reinforcement Learning Control - 100 Episodes
% This simulation models plasma confinement in a polywell reactor with
% RL-based magnetic field control through coil current adjustment

clear; clc; close all;

%% Physical Constants
e = 1.602e-19;          % Elementary charge (C)
m_e = 9.109e-31;        % Electron mass (kg)
m_p = 1.673e-27;        % Proton mass (kg)
mu_0 = 4*pi*1e-7;       % Permeability of free space (H/m)
k_B = 1.381e-23;        % Boltzmann constant (J/K)
epsilon_0 = 8.854e-12;  % Permittivity of free space (F/m)

%% Simulation Parameters - Optimized for 100 episodes
n_episodes = 100;       % Number of RL episodes
steps_per_episode = 50; % Time steps per episode
dt = 5e-9;              % Time step (s)
grid_size = 30;         % Reduced for faster computation
L = 0.1;                % Reactor size (m)
dx = L/(grid_size-1);   % Spatial resolution (m)

%% Polywell Geometry - 6 magnetic coils in cubic arrangement
n_coils = 6;
coil_radius = 0.02;     % Coil radius (m)
coil_positions = [
    L/2, 0, L/2;        % Front
    L/2, L, L/2;        % Back
    0, L/2, L/2;        % Left
    L, L/2, L/2;        % Right
    L/2, L/2, 0;        % Bottom
    L/2, L/2, L         % Top
];

%% Initial Plasma Parameters
n_particles = 200;      % Reduced for faster computation
T_e = 1e4;              % Electron temperature (K)
T_i = 1e4;              % Ion temperature (K)
n_density = 1e18;       % Plasma density (m^-3)

%% Reinforcement Learning Setup
n_actions = 5;          % Simplified action space
action_space = linspace(100, 1000, n_actions); % Current range (A)
n_states = 20;          % Simplified state space
alpha = 0.15;           % Learning rate
gamma = 0.9;            % Discount factor
epsilon = 0.3;          % Initial exploration rate
epsilon_decay = 0.995;  % Exploration decay
Q_table = zeros(n_states, n_actions);

% State discretization
state_bins = linspace(0, 1, n_states+1);

%% Episode Results Storage
episode_rewards = zeros(n_episodes, 1);
episode_confinement = zeros(n_episodes, 1);
episode_energy = zeros(n_episodes, 1);
best_currents = zeros(n_episodes, n_coils);
convergence_data = zeros(n_episodes, 3); % [mean_reward, max_confinement, exploration_rate]

fprintf('Starting polywell RL simulation - 100 episodes\n');
fprintf('Particles: %d, Grid: %dx%dx%d, Steps/episode: %d\n', ...
        n_particles, grid_size, grid_size, grid_size, steps_per_episode);
fprintf('Episode | Reward | Confinement | Energy | Exploration\n');
fprintf('--------|--------|-------------|--------|------------\n');

%% Main Episode Loop
for episode = 1:n_episodes
    % Initialize particles for this episode
    particles = initializeParticles(n_particles, L, T_e, T_i, m_e, m_p, e, k_B);
    
    % Initialize episode variables
    episode_reward = 0;
    coil_currents = ones(n_coils, 1) * 500; % Start with moderate currents
    
    % Run episode
    for step = 1:steps_per_episode
        %% Calculate Magnetic Field from Current Configuration
        [Bx, By, Bz] = calculateMagneticField(coil_positions, coil_currents, ...
                                              coil_radius, grid_size, L, mu_0);
        
        %% Update Particles
        particles = updateParticlesBoris(particles, Bx, By, Bz, dt, L, grid_size);
        
        %% Calculate State and Reward (every 10 steps)
        if mod(step, 10) == 0
            confinement_ratio = calculateConfinement(particles, L);
            plasma_energy = calculatePlasmaEnergy(particles);
            
            % Get current state
            state = discretizeState(confinement_ratio, state_bins);
            
            % Calculate reward
            reward = calculateReward(confinement_ratio, plasma_energy);
            episode_reward = episode_reward + reward;
            
            % RL Action Selection and Learning
            if step > 10 % Need previous state for Q-learning
                % Update Q-table
                Q_table(prev_state, prev_action) = Q_table(prev_state, prev_action) + ...
                    alpha * (prev_reward + gamma * max(Q_table(state, :)) - ...
                            Q_table(prev_state, prev_action));
            end
            
            % Epsilon-greedy action selection
            if rand < epsilon
                action = randi(n_actions);
            else
                [~, action] = max(Q_table(state, :));
            end
            
            % Apply action to coil currents
            base_current = action_space(action);
            coil_currents = generateCoilCurrents(base_current, n_coils, episode);
            
            % Store for next iteration
            prev_state = state;
            prev_action = action;
            prev_reward = reward;
        end
    end
    
    %% Episode Results
    final_confinement = calculateConfinement(particles, L);
    final_energy = calculatePlasmaEnergy(particles);
    
    % Store episode results
    episode_rewards(episode) = episode_reward;
    episode_confinement(episode) = final_confinement;
    episode_energy(episode) = final_energy;
    best_currents(episode, :) = coil_currents';
    convergence_data(episode, :) = [episode_reward, final_confinement, epsilon];
    
    % Decay exploration rate
    epsilon = epsilon * epsilon_decay;
    
    % Display progress every 10 episodes
    if mod(episode, 10) == 0
        fprintf('%7d | %6.3f | %11.3f | %6.2e | %10.3f\n', ...
                episode, episode_reward, final_confinement, final_energy, epsilon);
    end
end

%% Results Analysis and Visualization
fprintf('\nSimulation Complete!\n');
fprintf('Best confinement ratio: %.3f (Episode %d)\n', ...
        max(episode_confinement), find(episode_confinement == max(episode_confinement), 1));
fprintf('Average final 10 episodes confinement: %.3f\n', ...
        mean(episode_confinement(end-9:end)));

% Create comprehensive results visualization
figure('Position', [50, 50, 1400, 1000]);

% Episode rewards
subplot(2,4,1);
plot(1:n_episodes, episode_rewards, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:n_episodes, smooth(episode_rewards, 10), 'r-', 'LineWidth', 2);
xlabel('Episode'); ylabel('Total Reward');
title('Learning Progress - Rewards');
legend('Raw', 'Smoothed', 'Location', 'best');
grid on;

% Confinement evolution
subplot(2,4,2);
plot(1:n_episodes, episode_confinement, 'g-', 'LineWidth', 1.5);
hold on;
plot(1:n_episodes, smooth(episode_confinement, 10), 'k-', 'LineWidth', 2);
xlabel('Episode'); ylabel('Confinement Ratio');
title('Plasma Confinement Evolution');
legend('Raw', 'Smoothed', 'Location', 'best');
grid on;

% Energy evolution
subplot(2,4,3);
semilogy(1:n_episodes, episode_energy, 'm-', 'LineWidth', 1.5);
hold on;
semilogy(1:n_episodes, smooth(episode_energy, 10), 'c-', 'LineWidth', 2);
xlabel('Episode'); ylabel('Plasma Energy (J)');
title('Energy Evolution (Log Scale)');
legend('Raw', 'Smoothed', 'Location', 'best');
grid on;

% Exploration rate
subplot(2,4,4);
plot(1:n_episodes, convergence_data(:,3), 'r--', 'LineWidth', 2);
xlabel('Episode'); ylabel('Exploration Rate (ε)');
title('Exploration Decay');
grid on;

% Q-table heatmap
subplot(2,4,5);
imagesc(Q_table);
colorbar;
xlabel('Actions'); ylabel('States');
title('Q-Table Heatmap');
colormap(jet);

% Best coil currents evolution
subplot(2,4,6);
plot(1:n_episodes, best_currents, 'LineWidth', 1.5);
xlabel('Episode'); ylabel('Current (A)');
title('Optimal Coil Currents');
legend('Coil 1', 'Coil 2', 'Coil 3', 'Coil 4', 'Coil 5', 'Coil 6', 'Location', 'best');
grid on;

% Final particle positions (best episode)
[~, best_episode] = max(episode_confinement);
particles_final = initializeParticles(n_particles, L, T_e, T_i, m_e, m_p, e, k_B);
% Simulate best episode configuration
best_coil_currents = best_currents(best_episode, :)';
[Bx, By, Bz] = calculateMagneticField(coil_positions, best_coil_currents, ...
                                      coil_radius, grid_size, L, mu_0);
for i = 1:steps_per_episode
    particles_final = updateParticlesBoris(particles_final, Bx, By, Bz, dt, L, grid_size);
end

subplot(2,4,7:8);
scatter3(particles_final.x(:,1)*1000, particles_final.x(:,2)*1000, particles_final.x(:,3)*1000, ...
         30, particles_final.charge/e, 'filled', 'MarkerEdgeColor', 'k');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title(sprintf('Best Configuration Particles (Episode %d)', best_episode));
colorbar; 
caxis([-1, 1]);
colormap(gca, [0 0 1; 1 0 0]); % Blue for electrons, red for ions
axis equal; grid on; view(45, 30);

% Print final statistics
fprintf('\nFinal Q-Table Statistics:\n');
fprintf('Max Q-value: %.4f\n', max(Q_table(:)));
fprintf('Min Q-value: %.4f\n', min(Q_table(:)));
fprintf('Q-table sparsity: %.1f%% zeros\n', 100*sum(Q_table(:)==0)/numel(Q_table));

%% Supporting Functions

function particles = initializeParticles(n_particles, L, T_e, T_i, m_e, m_p, e, k_B)
    % Initialize particle ensemble
    particles.x = rand(n_particles, 3) * L * 0.6 + L * 0.2; % Start in central region
    particles.v = zeros(n_particles, 3);
    particles.charge = [ones(n_particles/2, 1) * (-e); ones(n_particles/2, 1) * e];
    particles.mass = [ones(n_particles/2, 1) * m_e; ones(n_particles/2, 1) * m_p];
    
    % Maxwell-Boltzmann velocity distribution
    for i = 1:n_particles
        if particles.mass(i) == m_e
            v_th = sqrt(2*k_B*T_e/m_e);
        else
            v_th = sqrt(2*k_B*T_i/m_p);
        end
        particles.v(i,:) = randn(1,3) * v_th/sqrt(3);
    end
end

function [Bx, By, Bz] = calculateMagneticField(coil_positions, currents, ...
                                               coil_radius, grid_size, L, mu_0)
    % Calculate magnetic field using simplified model
    [X, Y, Z] = meshgrid(linspace(0, L, grid_size));
    
    Bx = zeros(size(X));
    By = zeros(size(Y));
    Bz = zeros(size(Z));
    
    for coil = 1:size(coil_positions, 1)
        [dBx, dBy, dBz] = magneticDipole(X, Y, Z, coil_positions(coil, :), ...
                                         currents(coil), coil_radius, mu_0);
        Bx = Bx + dBx;
        By = By + dBy;
        Bz = Bz + dBz;
    end
end

function [Bx, By, Bz] = magneticDipole(X, Y, Z, pos, current, radius, mu_0)
    % Magnetic dipole field
    dx = X - pos(1);
    dy = Y - pos(2);
    dz = Z - pos(3);
    r = sqrt(dx.^2 + dy.^2 + dz.^2);
    r(r < radius/5) = radius/5; % Avoid singularities
    
    m = current * pi * radius^2; % Magnetic moment
    
    r3 = r.^3;
    r5 = r.^5;
    
    Bx = (mu_0*m/(4*pi)) * (3*dx.*dz./r5);
    By = (mu_0*m/(4*pi)) * (3*dy.*dz./r5);
    Bz = (mu_0*m/(4*pi)) * ((3*dz.^2./r5) - (1./r3));
end

function particles = updateParticlesBoris(particles, Bx, By, Bz, dt, L, grid_size)
    % Boris pusher algorithm
    n_particles = size(particles.x, 1);
    
    for i = 1:n_particles
        B_local = interpolateField(particles.x(i,:), Bx, By, Bz, L, grid_size);
        
        q = particles.charge(i);
        m = particles.mass(i);
        v = particles.v(i,:);
        
        % Boris algorithm
        t = (q*dt/(2*m)) * B_local;
        t_mag = norm(t);
        
        if t_mag > 1e-10
            s = 2*t/(1 + t_mag^2);
            v_prime = v + cross(v, t);
            particles.v(i,:) = v + cross(v_prime, s);
        end
        
        % Position update
        particles.x(i,:) = particles.x(i,:) + particles.v(i,:) * dt;
        
        % Boundary handling
        for dim = 1:3
            if particles.x(i,dim) < 0 || particles.x(i,dim) > L
                % Reflect and reduce energy
                particles.x(i,dim) = max(0, min(L, particles.x(i,dim)));
                particles.v(i,dim) = -0.8 * particles.v(i,dim);
            end
        end
    end
end

function B_local = interpolateField(pos, Bx, By, Bz, L, grid_size)
    % Nearest neighbor interpolation
    i = round(pos(1)/L * (grid_size-1)) + 1;
    j = round(pos(2)/L * (grid_size-1)) + 1;
    k = round(pos(3)/L * (grid_size-1)) + 1;
    
    i = max(1, min(grid_size, i));
    j = max(1, min(grid_size, j));
    k = max(1, min(grid_size, k));
    
    B_local = [Bx(j,i,k), By(j,i,k), Bz(j,i,k)];
end

function confinement_ratio = calculateConfinement(particles, L)
    % Confinement metric
    center = L/2;
    confinement_radius = L/3;
    
    distances = sqrt(sum((particles.x - center).^2, 2));
    confined = sum(distances < confinement_radius);
    confinement_ratio = confined / size(particles.x, 1);
end

function energy = calculatePlasmaEnergy(particles)
    % Total kinetic energy
    kinetic_energy = 0.5 * particles.mass .* sum(particles.v.^2, 2);
    energy = sum(kinetic_energy);
end

function reward = calculateReward(confinement_ratio, plasma_energy)
    % Multi-objective reward function
    confinement_reward = 10 * confinement_ratio^2;
    stability_reward = 5 * (1 - min(1, plasma_energy/1e-14));
    reward = confinement_reward + stability_reward;
end

function state = discretizeState(confinement_ratio, state_bins)
    % Discretize state
    state = max(1, min(length(state_bins)-1, ...
                      find(confinement_ratio >= state_bins, 1, 'last')));
end

function currents = generateCoilCurrents(base_current, n_coils, episode)
    % Generate coil current configuration
    currents = ones(n_coils, 1) * base_current;
    
    % Add learned asymmetry based on episode progress
    phase_shift = 2*pi/n_coils;
    for i = 1:n_coils
        variation = 0.2 * base_current * sin(episode/10 + i*phase_shift);
        currents(i) = max(50, currents(i) + variation);
    end
end

function smoothed = smooth(data, window_size)
    % Simple moving average smoother
    smoothed = data;
    for i = window_size:length(data)
        smoothed(i) = mean(data(i-window_size+1:i));
    end
end