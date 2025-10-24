%% MATLAB Polywell - Simplified Test Version
% Run this first to test if magnetic field calculation works

clear; clc; close all;

%% Parameters
coil_radius = 0.05;        % Coil radius (m)
chamber_radius = 0.30;     % Chamber radius (m)
coil_current = 5000;       % Current per coil (A)
coil_turns = 100;          % Number of turns per coil
mu0 = 4*pi*1e-7;          % Permeability of free space

%% Simple grid (smaller for testing)
nx = 21; ny = 21; nz = 21;  % Smaller grid for speed
x = linspace(-0.4, 0.4, nx);
y = linspace(-0.4, 0.4, ny);
z = linspace(-0.4, 0.4, nz);

fprintf('=== POLYWELL SIMPLE TEST ===\n');
fprintf('Grid size: %dx%dx%d\n', nx, ny, nz);

%% Coil geometry
d = chamber_radius * 0.7;
coil_pos = [
     d,  0,  0;   % +X coil
    -d,  0,  0;   % -X coil
     0,  d,  0;   % +Y coil
     0, -d,  0;   % -Y coil
     0,  0,  d;   % +Z coil
     0,  0, -d    % -Z coil
];

coil_normals = [
    -1,  0,  0;   % +X coil normal
     1,  0,  0;   % -X coil normal
     0, -1,  0;   % +Y coil normal
     0,  1,  0;   % -Y coil normal
     0,  0, -1;   % +Z coil normal
     0,  0,  1    % -Z coil normal
];

fprintf('Coil positions:\n');
for i = 1:size(coil_pos, 1)
    fprintf('  Coil %d: (%.2f, %.2f, %.2f)\n', i, coil_pos(i,:));
end

%% Simplified magnetic field calculation
fprintf('\nCalculating magnetic field (simplified)...\n');

[X, Y, Z] = meshgrid(x, y, z);
Bx = zeros(size(X));
By = zeros(size(Y)); 
Bz = zeros(size(Z));

total_current = coil_current * coil_turns;

for coil_idx = 1:size(coil_pos, 1)
    fprintf('Computing field from coil %d/%d...\n', coil_idx, size(coil_pos, 1));
    
    center = coil_pos(coil_idx, :);
    normal = coil_normals(coil_idx, :);
    
    % Simplified dipole approximation for faster calculation
    % Magnetic dipole moment
    m = total_current * pi * coil_radius^2;
    
    % Vector from coil to field points
    rx = X - center(1);
    ry = Y - center(2);
    rz = Z - center(3);
    r_mag = sqrt(rx.^2 + ry.^2 + rz.^2);
    
    % Avoid singularity
    r_mag(r_mag < 1e-6) = 1e-6;
    
    % Dipole field (simplified)
    % For a dipole aligned with normal direction
    dot_product = rx.*normal(1) + ry.*normal(2) + rz.*normal(3);
    
    factor = (mu0*m)/(4*pi) ./ (r_mag.^5);
    
    Bx = Bx + factor .* (3*dot_product.*rx - normal(1).*(r_mag.^2));
    By = By + factor .* (3*dot_product.*ry - normal(2).*(r_mag.^2));
    Bz = Bz + factor .* (3*dot_product.*rz - normal(3).*(r_mag.^2));
end

B_mag = sqrt(Bx.^2 + By.^2 + Bz.^2);

fprintf('Magnetic field calculation complete!\n');
fprintf('Max B field: %.3f T\n', max(B_mag(:)));
fprintf('Min B field: %.6f T\n', min(B_mag(:)));

%% Simple plasma parameters (no particle tracking)
plasma_density = 1e19;
density_e = plasma_density * exp(-B_mag / 0.1);  % Simple exponential profile
density_i = density_e;  % Quasi-neutrality
potential = 1000 * log(density_e / plasma_density);  % Simple potential

%% Simple fusion rate
Ti_keV = 10;  % 10 keV
sigma_DT = 5e-28;  % m² (simplified)
v_rel = 1e6;  % m/s (simplified)
fusion_rate = 0.25 * density_i.^2 * sigma_DT * v_rel;
fusion_power = fusion_rate * 17.6e6 * 1.602e-19;  % 17.6 MeV per reaction

%% Export data
fprintf('\nExporting data...\n');

[X, Y, Z] = meshgrid(x, y, z);
coords = [X(:), Y(:), Z(:)];
B_field = [Bx(:), By(:), Bz(:)];
plasma_data = [density_e(:), density_i(:), potential(:), fusion_rate(:), fusion_power(:)];

% Export CSV files
csvwrite('polywell_coordinates.csv', coords);
csvwrite('polywell_magnetic_field.csv', B_field);
csvwrite('polywell_plasma_data.csv', plasma_data);

% Export coil geometry
coil_data = [coil_pos, coil_normals, ...
             repmat([coil_radius, coil_current, coil_turns], size(coil_pos, 1), 1)];
csvwrite('polywell_coil_geometry.csv', coil_data);

fprintf('Files exported:\n');
fprintf('- polywell_coordinates.csv\n');
fprintf('- polywell_magnetic_field.csv\n');
fprintf('- polywell_plasma_data.csv\n');
fprintf('- polywell_coil_geometry.csv\n');

%% Visualization
fprintf('\nCreating plots...\n');

figure('Position', [100, 100, 1200, 800]);

% Magnetic field in XY plane (midplane)
subplot(2, 3, 1);
mid_z = ceil(nz/2);
[X_2D, Y_2D] = meshgrid(x, y);
B_slice = squeeze(B_mag(:, :, mid_z));
contourf(X_2D, Y_2D, B_slice', 20);
colorbar;
title('Magnetic Field |B| (T)');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on;

% Add coil positions
hold on;
for i = 1:size(coil_pos, 1)
    if abs(coil_pos(i, 3)) < 0.05  % Show midplane coils
        viscircles([coil_pos(i, 1), coil_pos(i, 2)], coil_radius, ...
                   'Color', 'r', 'LineWidth', 2);
    end
end

% Magnetic field vectors (sampled)
subplot(2, 3, 2);
skip = 2;  % Show every 2nd vector
x_sample = x(1:skip:end);
y_sample = y(1:skip:end);
[X_vec, Y_vec] = meshgrid(x_sample, y_sample);
Bx_sample = Bx(1:skip:end, 1:skip:end, mid_z);
By_sample = By(1:skip:end, 1:skip:end, mid_z);
quiver(X_vec, Y_vec, Bx_sample', By_sample', 2);
title('B-field Vectors (XY plane)');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on;

% Electron density
subplot(2, 3, 3);
density_slice = squeeze(density_e(:, :, mid_z));
contourf(X_2D, Y_2D, density_slice', 20);
colorbar;
title('Electron Density (m^{-3})');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on;

% Electric potential
subplot(2, 3, 4);
potential_slice = squeeze(potential(:, :, mid_z));
contourf(X_2D, Y_2D, potential_slice', 20);
colorbar;
title('Electric Potential (V)');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on;

% Fusion power
subplot(2, 3, 5);
fusion_slice = squeeze(fusion_power(:, :, mid_z));
contourf(X_2D, Y_2D, fusion_slice', 20);
colorbar;
title('Fusion Power (W/m³)');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on;

% 1D profiles along X-axis
subplot(2, 3, 6);
mid_y = ceil(ny/2);
x_profile = squeeze(B_mag(mid_y, :, mid_z));
plot(x, x_profile, 'b-', 'LineWidth', 2);
xlabel('X (m)');
ylabel('|B| (T)');
title('B-field profile along X-axis');
grid on;

sgtitle('Polywell Simulation Results');
saveas(gcf, 'polywell_simple_results.png');

%% Summary
fprintf('\n=== SIMULATION SUMMARY ===\n');
fprintf('Grid points: %d\n', length(coords));
fprintf('Max magnetic field: %.3f T\n', max(B_mag(:)));
fprintf('Min magnetic field: %.6f T\n', min(B_mag(:)));
fprintf('Average density: %.2e m^-3\n', mean(density_e(:)));
fprintf('Total fusion power: %.2e W\n', trapz(x, trapz(y, trapz(z, fusion_power))));

fprintf('\n✅ SIMPLE TEST COMPLETE!\n');
fprintf('Ready for ANSYS import!\n');