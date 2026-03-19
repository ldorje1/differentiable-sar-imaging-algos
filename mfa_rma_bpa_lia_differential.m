clc; clear; close all;

%% ========================================================================
% Differential Imaging Attack (DIA) for Near-Field SAR / mmWave Imaging
%
% This script performs an end-to-end differentiable attack on SAR data by
% optimizing a complex gain vector A across aperture locations.
%
% High-level flow:
%   1) Load victim SAR raw data and reconstruct the clean image.
%   2) Build a target image (either shuffled/noise-like or another object).
%   3) Build attack waveforms from X_aa and align them to the victim bin.
%   4) Optimize complex gains A with gradient descent.
%   5) Reconstruct the attacked image and report metrics.

%% ------------------------------------------------------------------------
% User selection
% ------------------------------------------------------------------------
clc; clear; close all;

dataDir     = fullfile(pwd, 'data');          % folder with .mat support files
raw_dataDir = fullfile(pwd, 'raw_sar_data');  % folder with raw SAR cubes

% save_final_images = true;   % set false to skip saving figures

%% ------------------------------------------------------------------------
% Step 1: Load victim data and define imaging parameters
% ------------------------------------------------------------------------
sar_algo    = 'MFA';     % options: 'MFA', 'RMA', 'BPA', 'LIA'
target_mode = 'object';  % options: 'noise' or 'object'

target_raw_select = 5;   % object target index (1..10), only used in object mode
rawData_select    = 1;   % victim raw cube index

rawData = [ ...
    "knife", ...               % 1
    "plier", ...               % 2
    "scissor", ...             % 3
    "screw_driver", ...        % 4
    "sharp_paint_speader", ... % 5
    "dragger", ...             % 6
    "wrench", ...              % 7
    "gun", ...                 % 8
    "rifle", ...               % 9
    "butcher_knife" ...        % 10
];

% -------------------------------------------------------------------------
% Victim geometry and sampling parameters
% -------------------------------------------------------------------------
switch lower(rawData(rawData_select))
    case "knife",                dx = 1; dy = 1; z0 = 185; FS = 5000e3;
    case "plier",                dx = 1; dy = 2; z0 = 210; FS = 5000e3;
    case "scissor",              dx = 1; dy = 2; z0 = 215; FS = 5000e3;
    case "screw_driver",         dx = 1; dy = 2; z0 = 230; FS = 5000e3;
    case "sharp_paint_speader",  dx = 1; dy = 2; z0 = 180; FS = 5000e3;
    case "dragger",              dx = 1; dy = 2; z0 = 195; FS = 5000e3;
    case "wrench",               dx = 1; dy = 1; z0 = 170; FS = 9121e3;
    case "gun",                  dx = 1; dy = 1; z0 = 185; FS = 9121e3;
    case "rifle",                dx = 1; dy = 1; z0 = 185; FS = 9121e3;
    case "butcher_knife",        dx = 1; dy = 2; z0 = 210; FS = 5000e3;
    otherwise
        error("Unknown rawData selection: %s", rawData(rawData_select));
end

% -------------------------------------------------------------------------
% Target geometry and sampling parameters (object target only)
% -------------------------------------------------------------------------
if strcmpi(target_mode, 'object')
    switch lower(rawData(target_raw_select))
        case "knife",                dx_t = 1; dy_t = 1; z0_tgt = 185; FS_tgt = 5000e3;
        case "plier",                dx_t = 1; dy_t = 2; z0_tgt = 220; FS_tgt = 5000e3;
        case "scissor",              dx_t = 1; dy_t = 2; z0_tgt = 215; FS_tgt = 5000e3;
        case "screw_driver",         dx_t = 1; dy_t = 2; z0_tgt = 230; FS_tgt = 5000e3;
        case "sharp_paint_speader",  dx_t = 1; dy_t = 2; z0_tgt = 180; FS_tgt = 5000e3;
        case "dragger",              dx_t = 1; dy_t = 2; z0_tgt = 195; FS_tgt = 5000e3;
        case "wrench",               dx_t = 1; dy_t = 1; z0_tgt = 180; FS_tgt = 9121e3;
        case "gun",                  dx_t = 1; dy_t = 1; z0_tgt = 185; FS_tgt = 9121e3;
        case "rifle",                dx_t = 1; dy_t = 1; z0_tgt = 185; FS_tgt = 9121e3;
        case "butcher_knife",        dx_t = 1; dy_t = 2; z0_tgt = 210; FS_tgt = 5000e3;
        otherwise
            error("Unknown target_raw_select: %s", rawData(target_raw_select));
    end
end

c0 = physconst('lightspeed');
F0 = 77e9;         % FMCW start frequency (Hz)
K0 = 70.295e12;    % chirp slope (Hz/s)
tI = 4.5225e-10;   % instrument delay (s)

nFFTtime  = 1024;  % FFT size along fast time / range
nFFTspace = 1024;  % FFT size in spatial domain (used by MFA/RMA)

% -------------------------------------------------------------------------
% Load victim and target raw SAR cubes
% -------------------------------------------------------------------------
sarRawData = load(fullfile(raw_dataDir, rawData(rawData_select) + ".mat")).adcDataCube;

if strcmpi(target_mode, 'object')
    sarRawData_tgt = load(fullfile(raw_dataDir, rawData(target_raw_select) + ".mat")).adcDataCube;
end

[Nsamp, M, N] = size(sarRawData);

X_v = reshape(sarRawData, Nsamp, M * N);  % vectorized victim measurements
Np  = M * N;                              % number of aperture positions

%% ------------------------------------------------------------------------
% Step 2: Raw-data preprocessing and clean / target image generation
% ------------------------------------------------------------------------
switch upper(sar_algo)

    case 'MFA' % ==========================================================
        bbox = [-200 200 -200 200];  % [xmin xmax ymin ymax] in mm

        % Range-bin index from FMCW delay model
        k0_range_bin = round(K0 / FS * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);

        % FFT over fast time, then gate the selected range bin
        rawDataFFT = fft(sarRawData, nFFTtime);
        sarData    = squeeze(rawDataFFT(k0_range_bin + 1, :, :));  % M x N

        % Correct serpentine scan ordering
        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        % Imaging parameters used by the differentiable reconstructor
        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, ...
                        'z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo, ...
                        'target_mode', target_mode);

        % Clean reconstruction
        [~, ~, clean_img, ~] = dlMFA(sarData, params);
        clean_img = extractdata(clean_img);

        % Build target image
        if strcmpi(target_mode, 'noise')
            % Shuffle gated measurements to create a structured random target
            [rows, cols] = size(sarData);
            rng(42);
            sarData_shuffled = reshape(sarData(randperm(rows * cols)), rows, cols);

            [~, ~, target_img, ~] = dlMFA(sarData_shuffled, params);
            target_img = extractdata(target_img);

        elseif strcmpi(target_mode, 'object')
            % Build target with its own geometry so victim params stay unchanged
            params_t = params;
            params_t.dx = dx_t;
            params_t.dy = dy_t;
            params_t.z0 = z0_tgt;

            kbin_t = round(K0 / FS_tgt * (2 * z0_tgt * 1e-3 / c0 + tI) * nFFTtime);

            rawDataFFT_t = fft(sarRawData_tgt, nFFTtime);
            sarData_t    = squeeze(rawDataFFT_t(kbin_t + 1, :, :));

            for ii = 2:2:size(sarData_t, 1)
                sarData_t(ii, :) = fliplr(sarData_t(ii, :));
            end

            params_t.k0_range_bin = kbin_t;

            [~, ~, target_img, ~] = dlMFA(sarData_t, params_t);
            target_img = extractdata(target_img);

        else
            error("target_mode must be 'noise' or 'object'.");
        end

        % Resize target to match victim image size if needed
        if ~isequal(size(target_img), size(clean_img))
            fprintf('Resizing target_img %s -> %s to match clean_img grid.\n', ...
                mat2str(size(target_img)), mat2str(size(clean_img)));

            if exist('imresize', 'file') == 2
                target_img = imresize(target_img, size(clean_img), 'bilinear');
            else
                % Fallback without Image Processing Toolbox
                [X, Y]  = meshgrid(linspace(0,1,size(target_img,2)), linspace(0,1,size(target_img,1)));
                [Xq,Yq] = meshgrid(linspace(0,1,size(clean_img,2)),  linspace(0,1,size(clean_img,1)));
                target_img = interp2(X, Y, target_img, Xq, Yq, 'linear', 0);
            end
        end

    case 'RMA' % ==========================================================
        % Reorder cube to match the expected layout in the RMA implementation
        Echo = permute(sarRawData, [3, 2, 1]);  % -> [horizontal, vertical, samples]
        bbox = [-200 200 -200 200];

        [Nx, Nz, ~] = size(Echo);

        num_sample = size(Echo, 3);
        nFFTtime   = num_sample;
        rawDataFFT = fft(Echo, nFFTtime, 3);

        % Select the dominant range bin using total energy
        E = squeeze(sum(sum(abs(rawDataFFT).^2, 1), 2));
        [~, k0_range_bin] = max(E);

        % Manual overrides for cases where max-energy bin is unreliable
        label = lower(rawData(rawData_select));
        switch label
            case {"screw_driver", "gun", "rifle"}
                k0_range_bin = 8;
            case "wrench"
                k0_range_bin = 7;
        end

        % Gate the chosen bin and transpose to the expected shape
        sarData = squeeze(rawDataFFT(:, :, k0_range_bin)).';
        z0_t = (c0/2) * (((k0_range_bin - 1) / (K0 * (1 / FS) * nFFTtime)) - tI);

        % Correct serpentine scan ordering
        for ii = 2:2:Nz
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        % Imaging parameters
        params = struct('nFFTspace', nFFTspace, 'nFFTtime', nFFTtime, ...
                        'z0', z0_t, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'F0', F0, 'Nsamp', Nsamp, 'N', N, 'M', M, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo, ...
                        'target_mode', target_mode);

        % Clean reconstruction
        [~, ~, clean_img, ~] = dlRMA(sarData, params);
        clean_img = extractdata(clean_img);

        % Build target image
        if strcmpi(target_mode, 'noise')
            [rows, cols] = size(sarData);
            rng(42);
            sarData_shuffled = reshape(sarData(randperm(rows * cols)), rows, cols);

            [~, ~, target_img, ~] = dlRMA(sarData_shuffled, params);
            target_img = extractdata(target_img);

        elseif strcmpi(target_mode, 'object')
            Echo_t = permute(sarRawData_tgt, [3, 2, 1]);

            num_sample_t = size(Echo_t, 3);
            rawDataFFT_t = fft(Echo_t, num_sample_t, 3);

            % Select target bin by max energy
            E_t = squeeze(sum(sum(abs(rawDataFFT_t).^2, 1), 2));
            [~, kbin_t] = max(E_t);

            % Manual overrides for known bad cases
            label_t = lower(rawData(target_raw_select));
            switch label_t
                case {"screw_driver", "gun", "rifle"}
                    kbin_t = 8;
                case "wrench"
                    kbin_t = 7;
            end

            sarData_t = squeeze(rawDataFFT_t(:, :, kbin_t)).';

            [Nx_t, Nz_t, ~] = size(Echo_t);

            for ii = 2:2:Nz_t
                sarData_t(ii, :) = fliplr(sarData_t(ii, :));
            end

            z0_t = (c0/2) * (((kbin_t - 1) / (K0 * (1 / FS_tgt) * num_sample_t)) - tI);

            params_t = params;
            params_t.dx = dx_t;
            params_t.dy = dy_t;
            params_t.z0 = z0_t;
            params_t.nFFTtime = num_sample_t;
            params_t.k0_range_bin = kbin_t;

            [~, ~, target_img, ~] = dlRMA(sarData_t, params_t);
            target_img = extractdata(target_img);
        else
            error("target_mode must be 'noise' or 'object'.");
        end

        % Resize target to match victim image size if needed
        if ~isequal(size(target_img), size(clean_img))
            fprintf('Resizing target_img %s -> %s to match clean_img grid.\n', ...
                mat2str(size(target_img)), mat2str(size(clean_img)));

            if exist('imresize', 'file') == 2
                target_img = imresize(target_img, size(clean_img), 'bilinear');
            else
                [X, Y]  = meshgrid(linspace(0,1,size(target_img,2)), linspace(0,1,size(target_img,1)));
                [Xq,Yq] = meshgrid(linspace(0,1,size(clean_img,2)),  linspace(0,1,size(clean_img,1)));
                target_img = interp2(X, Y, target_img, Xq, Yq, 'linear', 0);
            end
        end

    case 'BPA' % ==========================================================
        bbox = [-200 200 -200 200];

        % FFT over fast time, then gate the selected range bin
        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * (1 / FS) * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);

        % Manual overrides for known cases
        label = lower(rawData(rawData_select));
        switch label
            case {"gun"}
                k0_range_bin = 14;
            case "rifle"
                k0_range_bin = 13;
        end

        sarData = squeeze(rawDataFFT(k0_range_bin + 1, :, :));

        % Correct serpentine scan ordering
        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        % BPA image size
        A = 50;
        B = 50;

        % BPA parameters
        params = struct('z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'Nsamp', Nsamp, 'nFFTtime', nFFTtime, 'N', N, 'M', M, ...
                        'A_bpa', A, 'B_bpa', B, 'F0', F0, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo, ...
                        'target_mode', target_mode);

        % Precompute BPA propagation matrix
        H_bpa        = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        % Clean reconstruction
        [~, ~, clean_img, ~] = dlBPA(sarData, params, H_bpa);
        clean_img = extractdata(clean_img);

        if strcmpi(target_mode, 'noise')
            [rows, cols] = size(sarData);
            rng(42);
            sarData_shuffled = reshape(sarData(randperm(rows * cols)), rows, cols);

            [~, ~, target_img, ~] = dlBPA(sarData_shuffled, params, H_bpa);
            target_img = extractdata(target_img);

        elseif strcmpi(target_mode, 'object')
            [Nsamp_t, M_t, N_t] = size(sarRawData_tgt);

            params_t = params;
            params_t.M     = M_t;
            params_t.N     = N_t;
            params_t.Nsamp = Nsamp_t;
            params_t.dx    = dx_t;
            params_t.dy    = dy_t;
            params_t.z0    = z0_tgt;

            kbin_t = round(K0 * (1 / FS_tgt) * (2 * z0_tgt * 1e-3 / c0 + tI) * nFFTtime);

            label_t = lower(rawData(target_raw_select));
            switch label_t
                case {"gun"}
                    kbin_t = 14;
                case "rifle"
                    kbin_t = 13;
            end

            rawDataFFT_t = fft(sarRawData_tgt, nFFTtime);
            sarData_t    = squeeze(rawDataFFT_t(kbin_t + 1, :, :));

            for ii = 2:2:size(sarData_t, 1)
                sarData_t(ii, :) = fliplr(sarData_t(ii, :));
            end

            params_t.k0_range_bin = kbin_t;

            % Recompute H if target geometry differs
            H_t = dlBPA_H_matrix(params_t);
            params_t.H_bpa = H_t;

            [~, ~, target_img, ~] = dlBPA(sarData_t, params_t, H_t);
            target_img = extractdata(target_img);

        else
            error("target_mode must be 'noise' or 'object'.");
        end

    case 'LIA' % ==========================================================
        bbox = [-200 200 -200 200];

        % FFT over fast time, then gate the selected range bin
        rawDataFFT   = fft(sarRawData, nFFTtime);
        k0_range_bin = round(K0 * (1 / FS) * (2 * z0 * 1e-3 / c0 + tI) * nFFTtime);

        label = lower(rawData(rawData_select));
        switch label
            case {"gun"}
                k0_range_bin = 14;
            case "rifle"
                k0_range_bin = 13;
        end

        sarData = squeeze(rawDataFFT(k0_range_bin + 1, :, :));

        % Correct serpentine scan ordering
        for ii = 2:2:size(sarData, 1)
            sarData(ii, :) = fliplr(sarData(ii, :));
        end

        % LIA image size
        A = 50;
        B = 50;

        % LIA parameters
        params = struct('z0', z0, 'dx', dx, 'dy', dy, 'bbox', bbox, ...
                        'Nsamp', Nsamp, 'nFFTtime', nFFTtime, 'N', N, 'M', M, ...
                        'A_bpa', A, 'B_bpa', B, 'F0', F0, ...
                        'k0_range_bin', k0_range_bin, 'sar_algo', sar_algo, ...
                        'target_mode', target_mode);

        % LIA reuses the BPA propagation matrix
        H_bpa        = dlBPA_H_matrix(params);
        params.H_bpa = H_bpa;

        % Random measurement subset used by LIA
        NM = M * N;
        kk = min(40000, NM);
        rng(1000);
        params.py = sort(randperm(NM, kk));

        % Clean reconstruction
        [~, ~, clean_img, ~] = dlLIA(sarData, params, H_bpa);
        clean_img = extractdata(clean_img);

        % Also reconstruct with BPA for scale consistency
        [~, ~, clean_img_2, ~] = dlBPA(sarData, params, H_bpa);
        clean_img_2 = extractdata(clean_img_2);

        % Build target image
        if strcmpi(target_mode, 'noise')
            [rows, cols] = size(sarData);
            rng(42);
            sarData_shuffled = reshape(sarData(randperm(rows * cols)), rows, cols);

            [~, ~, target_img, ~] = dlBPA(sarData_shuffled, params, H_bpa);
            target_img = extractdata(target_img);

        elseif strcmpi(target_mode, 'object')
            [Nsamp_t, M_t, N_t] = size(sarRawData_tgt);

            params_t = params;
            params_t.M     = M_t;
            params_t.N     = N_t;
            params_t.Nsamp = Nsamp_t;
            params_t.dx    = dx_t;
            params_t.dy    = dy_t;
            params_t.z0    = z0_tgt;

            kbin_t = round(K0 * (1 / FS_tgt) * (2 * z0_tgt * 1e-3 / c0 + tI) * nFFTtime);

            label_t = lower(rawData(target_raw_select));
            switch label_t
                case {"gun"}
                    kbin_t = 14;
                case "rifle"
                    kbin_t = 13;
            end

            rawDataFFT_t = fft(sarRawData_tgt, nFFTtime);
            sarData_t    = squeeze(rawDataFFT_t(kbin_t + 1, :, :));

            for ii = 2:2:size(sarData_t, 1)
                sarData_t(ii, :) = fliplr(sarData_t(ii, :));
            end

            params_t.k0_range_bin = kbin_t;

            % Recompute H if target geometry differs
            H_t = dlBPA_H_matrix(params_t);
            params_t.H_bpa = H_t;

            % Use the same subsampling strategy for target reconstruction
            NM_t = size(sarData_t, 1) * size(sarData_t, 2);
            kk_t = min(40000, NM_t);
            rng(1000);
            params_t.py = sort(randperm(NM_t, kk_t));

            % Target is reconstructed with BPA for scale consistency
            [~, ~, target_img, ~] = dlBPA(sarData_t, params_t, H_t);
            target_img = extractdata(target_img);

        else
            error("target_mode must be 'noise' or 'object'.");
        end

    otherwise
        error('Invalid SAR algorithm selection.');
end

%% ------------------------------------------------------------------------
% Quick visualization: clean vs. target
% ------------------------------------------------------------------------
figure();
subplot(1,2,1);
imagesc(clean_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
title(sprintf('Clean Image (%s)', upper(sar_algo)));
axis image off;

subplot(1,2,2);
imagesc(target_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
title(sprintf('Target Image (%s)', upper(sar_algo)));
axis image off;

fprintf('clean_img  abs min/max : %.6e / %.6e\n', ...
    min(abs(clean_img(:))), max(abs(clean_img(:))));

fprintf('target_img abs min/max : %.6e / %.6e\n', ...
    min(abs(target_img(:))), max(abs(target_img(:))));

%% ------------------------------------------------------------------------
% Normalize images and store dlarray versions in params
% ------------------------------------------------------------------------
if strcmpi(sar_algo, 'LIA')
    % LIA clean output is normalized with the LIA scale
    global_scale_lia = max(abs(clean_img(:)))   + 1e-12;

    % Target and attack objective are kept in BPA scale for consistency
    global_scale_bpa = max(abs(clean_img_2(:))) + 1e-12;

    params.global_scale_lia = global_scale_lia;
    params.global_scale_bpa = global_scale_bpa;

    clean_img  = clean_img  / global_scale_lia;
    target_img = target_img / global_scale_bpa;

else
    global_scale = max(abs(clean_img(:))) + 1e-12;
    params.global_scale = global_scale;

    clean_img  = clean_img  / global_scale;
    target_img = target_img / global_scale;
end

% Store dlarray versions for the optimization loss
params.clean_img  = dlarray(clean_img,  "SS");
params.target_img = dlarray(target_img, "SS");

fprintf('clean_img  abs min/max : %.6e / %.6e\n', ...
    min(abs(clean_img(:))), max(abs(clean_img(:))));

fprintf('target_img abs min/max : %.6e / %.6e\n', ...
    min(abs(target_img(:))), max(abs(target_img(:))));

%% ------------------------------------------------------------------------
% Step 3: Build attack waveforms and optimize complex gains A
% ------------------------------------------------------------------------

% Load attack waveform pool
temp_x_aa_1 = load(fullfile(dataDir, "X_aa.mat"));
X_aa_1 = temp_x_aa_1.X_aa;

temp_x_aa_2 = load(fullfile(dataDir, "X_aa_2.mat"));
X_aa_2 = temp_x_aa_2.X_aa;

%% ------------------------------------------------------------------------
% 3.1 Select attack waveforms and align them to the victim range bin
% ------------------------------------------------------------------------
if Nsamp == 512 & M * N > 40000
    X_aa_temp = [X_aa_1; X_aa_2];
    X_aa = [X_aa_temp, X_aa_temp];
elseif Nsamp == 512 & M * N == 40000
    X_aa = [X_aa_1; X_aa_2];
else
    X_aa = X_aa_1;
end

targetK = M * N;   % number of waveforms sampled from the pool
rng(0);            % fixed seed for reproducibility

% Sample random waveform columns from the attack pool
sample_idx = randi(size(X_aa, 2), [1, targetK]);
X_a_pool   = X_aa(:, sample_idx);

% Select one waveform per aperture position
sel_idx = randi(size(X_a_pool, 2), [1, Np]);
X_a     = X_a_pool(:, sel_idx);

% -------------------------------------------------------------------------
% Shift each waveform to align with the victim range bin
% -------------------------------------------------------------------------
Xspec = fft(X_a, nFFTtime, 1);
[~, b0] = max(abs(Xspec), [], 1);
f0    = (b0 - 1) * FS / nFFTtime;
f_tgt = (params.k0_range_bin) * FS / nFFTtime;
Delta = f0 - f_tgt;

t = (0:Nsamp-1).' / FS;
P = exp(-1j * 2 * pi * (t * Delta));
D = P .* X_a;

% -------------------------------------------------------------------------
% RMS-match each attack waveform to the corresponding victim measurement
% -------------------------------------------------------------------------
colrms = @(X) sqrt(mean(abs(X).^2, 1));
scale  = (colrms(X_v) + eps) ./ (colrms(D) + eps);
D      = D .* scale;

%% ------------------------------------------------------------------------
% 3.2 DIA optimization
% ------------------------------------------------------------------------
save_final_images = false;   % set false to skip saving figures

switch upper(sar_algo)
    case 'MFA'
        maxIter   = 400;
        lr_re     = 1e4;
        lr_im     = 1e4;
        lambda_L2 = 1e-5;

    case 'RMA'
        maxIter   = 1000;
        lr_re     = 1e4;
        lr_im     = 1e4;
        lambda_L2 = 1e-5;

    case 'BPA'
        maxIter   = 500;
        lr_re     = 1e3;
        lr_im     = 1e3;
        lambda_L2 = 1e-5;

    case 'LIA'
        maxIter   = 500;
        lr_re     = 1e4;
        lr_im     = 1e4;
        lambda_L2 = 1e-5;

    otherwise
        error('Invalid SAR algorithm selection.');
end

%% ------------------------------------------------------------------------
% Constraint settings
% ------------------------------------------------------------------------
use_pa_pr_projection = false;  % enforce Pa/Pr <= PaPr_max if true
PaPr_max_dB          = -10;
PaPr_max             = 10^(PaPr_max_dB / 10);

use_amax_projection  = false;  % enforce |A| <= Amax if true
Amax                 = 2;

% Reference power for Pa/Pr normalization
Pr_ref = norm(X_v, 'fro')^2 + 1e-12;

% Initialize complex gain vector A = A_re + j*A_im
A_re = dlarray(1e-3 * randn(Np, 1, 'double'));
A_im = dlarray(1e-3 * randn(Np, 1, 'double'));

% History arrays
loss_hist   = zeros(maxIter, 1);
meanA_hist  = zeros(maxIter, 1);
maxA_hist   = zeros(maxIter, 1);
PaPr_hist   = zeros(maxIter, 1);
PaPrdB_hist = zeros(maxIter, 1);

fprintf(['\nOptimizing complex gain A for all locations (Np=%d) with SAR algorithm: %s ' ...
         '| lr_re=%.3g, lr_im=%.3g | lambda_L2=%.3g | ' ...
         'Pa/Pr<=%.2f dB (%d) | |A|<=%.3g (%d)\n'], ...
        Np, params.sar_algo, lr_re, lr_im, lambda_L2, ...
        PaPr_max_dB, use_pa_pr_projection, Amax, use_amax_projection);

%% ------------------------------------------------------------------------
% Main optimization loop
%
% Goal:
%   Minimize image-domain loss with respect to A_re and A_im
%
% Optional projections:
%   - per-element amplitude cap |A| <= Amax
%   - total injected power ratio Pa/Pr <= PaPr_max
%
% Early stop:
%   Stop when improvement over a sliding window becomes too small
% ------------------------------------------------------------------------
maxIter_cap = maxIter;

% Diminishing-returns early-stop settings
check_every  = 25;
W            = 50;
rel_drop_min = 1e-2;
abs_drop_min = 0;
K_weak       = 3;
weakWinCount = 0;

% Keep-best tracking
bestLoss = inf;
bestIter = 0;
bestA_re = A_re;
bestA_im = A_im;

iter = 0;

while true
    if iter >= maxIter_cap
        fprintf('Reached maxIter cap (%d).\n', maxIter_cap);
        break;
    end

    iter = iter + 1;

    % Forward pass + gradients through the full reconstruction pipeline
    [loss, gRe, gIm, ~] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);

    lossVal         = double(gather(extractdata(loss)));
    loss_hist(iter) = lossVal;

    % Gradient descent update
    A_re = A_re - lr_re * gRe;
    A_im = A_im - lr_im * gIm;

    % Convert to numeric complex vector for projections and logging
    A_num = extractdata(A_re) + 1j * extractdata(A_im);

    % Optional projection: cap individual gain magnitude
    if use_amax_projection
        mags = abs(A_num);
        over = mags > Amax;
        if any(over)
            scale_amax       = ones(size(mags));
            scale_amax(over) = Amax ./ mags(over);
            A_num            = A_num .* scale_amax;
        end
    end

    % Optional projection: cap total attack power ratio
    delta_now = D .* (A_num.');
    Pa_now    = norm(delta_now, 'fro')^2;
    PaPr_now  = Pa_now / Pr_ref;

    if use_pa_pr_projection && (PaPr_now > PaPr_max)
        scale_pow = sqrt(PaPr_max / (PaPr_now + 1e-12));
        A_num     = A_num * scale_pow;

        % Recompute after projection for accurate logging
        delta_now = D .* (A_num.');
        Pa_now    = norm(delta_now, 'fro')^2;
        PaPr_now  = Pa_now / Pr_ref;
    end
    PaPr_now_dB = 10 * log10(PaPr_now + 1e-12);

    % Write projected values back into dlarray form
    A_re = dlarray(real(A_num));
    A_im = dlarray(imag(A_num));

    % Record stats
    meanA_now = mean(abs(A_num), 'all');
    maxA_now  = max(abs(A_num), [], 'all');

    meanA_hist(iter)  = meanA_now;
    maxA_hist(iter)   = maxA_now;
    PaPr_hist(iter)   = PaPr_now;
    PaPrdB_hist(iter) = PaPr_now_dB;

    % Keep the best solution seen so far
    if lossVal < bestLoss
        bestLoss = lossVal;
        bestIter = iter;
        bestA_re = A_re;
        bestA_im = A_im;
    end

    % Progress logging
    if mod(iter, 25) == 0 || iter == 1 || iter == maxIter_cap
        [loss_log, ~, ~, atkImage_log] = dlfeval(@loss_and_grad, X_v, D, A_re, A_im, params, lambda_L2);
        lossVal_log = double(gather(extractdata(loss_log)));

        mse_AT = double(gather(extractdata(mean((atkImage_log - params.target_img).^2, 'all'))));
        mse_CA = double(gather(extractdata(mean((atkImage_log - params.clean_img).^2, 'all'))));

        gRe_num = extractdata(gRe);
        gIm_num = extractdata(gIm);
        Gnow    = max([max(abs(gRe_num), [], 'all'), max(abs(gIm_num), [], 'all')]);

        fprintf(['Iter %04d | Loss=%.4e (best=%.4e @%d) | G=%.6e | ' ...
                 'MSE(A,T)=%.4e, MSE(C,A)=%.4e | E|A|=%.4e, max|A|=%.4e | ' ...
                 'Pa/Pr=%.4e (%.2f dB)\n'], ...
                iter, lossVal_log, bestLoss, bestIter, Gnow, ...
                mse_AT, mse_CA, meanA_now, maxA_now, PaPr_now, PaPr_now_dB);
    end

    % Early stop based on weak improvement over a sliding window
    if iter > W && mod(iter, check_every) == 0
        L_old = loss_hist(iter - W);
        L_new = loss_hist(iter);

        rel_drop = (L_old - L_new) / max(abs(L_old), 1e-12);
        abs_drop = (L_old - L_new);

        if abs_drop_min > 0
            isWeak = (rel_drop < rel_drop_min) && (abs_drop < abs_drop_min);
        else
            isWeak = (rel_drop < rel_drop_min);
        end

        if isWeak
            weakWinCount = weakWinCount + 1;
        else
            weakWinCount = 0;
        end

        if weakWinCount >= K_weak
            fprintf(['Early stop: diminishing returns. Over last %d iters: ' ...
                     'rel_drop=%.3e, abs_drop=%.3e. Triggered %d/%d.\n'], ...
                     W, rel_drop, abs_drop, weakWinCount, K_weak);
            break;
        end
    end
end

% Trim histories to the actual number of iterations
loss_hist   = loss_hist(1:iter);
meanA_hist  = meanA_hist(1:iter);
maxA_hist   = maxA_hist(1:iter);
PaPr_hist   = PaPr_hist(1:iter);
PaPrdB_hist = PaPrdB_hist(1:iter);

% Restore the best solution before final evaluation
A_re = bestA_re;
A_im = bestA_im;

fprintf('-----------------------------\n');
fprintf('Done. Best loss %.4e at iter %d (ran %d iters).\n', bestLoss, bestIter, iter);
fprintf('-----------------------------\n');

%% ------------------------------------------------------------------------
% Step 4: Final evaluation
% ------------------------------------------------------------------------

% Reconstruct attacked measurements using the optimized gain vector
A_opt  = extractdata(A_re) + 1j * extractdata(A_im);
Y_opt  = X_v + D .* (A_opt.');
Y_cube = reshape(Y_opt, Nsamp, M, N);

% Gate the same range bin used during clean/target generation
rawDataFFT_att = fft(Y_cube, nFFTtime);
sarData_att    = squeeze(rawDataFFT_att(k0_range_bin + 1, :, :));

% Correct serpentine scan ordering
for ii = 2:2:size(sarData_att, 1)
    sarData_att(ii, :) = fliplr(sarData_att(ii, :));
end

% Final attacked reconstruction
switch upper(params.sar_algo)
    case 'MFA'
        [~, ~, atkImage_abs, ~] = dlMFA(sarData_att, params);
        adv_img = gather(extractdata(atkImage_abs));

    case 'RMA'
        [~, ~, atkImage_abs, ~] = dlRMA(sarData_att, params);
        adv_img = gather(extractdata(atkImage_abs));

    case 'BPA'
        [~, ~, atkImage_abs, ~] = dlBPA(sarData_att, params, params.H_bpa);
        adv_img = gather(extractdata(atkImage_abs));

    case 'LIA'
        [~, ~, atkImage_abs, ~] = dlLIA(sarData_att, params, params.H_bpa);
        adv_img = gather(extractdata(atkImage_abs));

    otherwise
        error('Invalid SAR algorithm selection for final reconstruction.');
end

%% ------------------------------------------------------------------------
% Step 5: Final metrics
% ------------------------------------------------------------------------
if strcmpi(params.sar_algo, 'LIA')
    adv_img = adv_img / params.global_scale_lia;
else
    adv_img = adv_img / params.global_scale;
end

% MSE
mse_AT = mean((adv_img(:) - target_img(:)).^2);
mse_AC = mean((adv_img(:) - clean_img(:)).^2);

% NCC
num    = sum(adv_img(:) .* target_img(:));
den    = sqrt(sum(adv_img(:).^2) * sum(target_img(:).^2)) + 1e-12;
ncc_AT = num / den;

num_AC = sum(adv_img(:) .* clean_img(:));
den_AC = sqrt(sum(adv_img(:).^2) * sum(clean_img(:).^2)) + 1e-12;
ncc_AC = num_AC / den_AC;

% Dynamic range
data_range_T = (max(target_img(:)) - min(target_img(:))) + 1e-12;
data_range_C = (max(clean_img(:))  - min(clean_img(:)))  + 1e-12;

% SSIM
if exist('ssim', 'file') == 2
    ssim_AT = ssim(adv_img, target_img, 'DynamicRange', data_range_T);
    ssim_AC = ssim(adv_img, clean_img,  'DynamicRange', data_range_C);
else
    ssim_AT = NaN;
    ssim_AC = NaN;
end

% PSNR
if exist('psnr', 'file') == 2
    psnr_AT = psnr(adv_img, target_img, data_range_T);
    psnr_AC = psnr(adv_img, clean_img,  data_range_C);
else
    psnr_AT = 10 * log10((data_range_T^2) / (mse_AT + 1e-12));
    psnr_AC = 10 * log10((data_range_C^2) / (mse_AC + 1e-12));
end

% Final signal-domain power ratio Pa/Pr
delta_opt     = D .* (A_opt.');
Pa_final      = norm(delta_opt, 'fro')^2;
Pr_final      = norm(X_v, 'fro')^2 + 1e-12;
PaPr_final    = Pa_final / Pr_final;
PaPr_final_dB = 10 * log10(PaPr_final + 1e-12);

% Print summary
fprintf('\n--- %s SR-level attack metrics ---\n', upper(params.sar_algo));
fprintf('MSE(A,T)      : %.4e\n', mse_AT);
fprintf('MSE(A,C)      : %.4e\n', mse_AC);
fprintf('NCC(A,T)      : %.4f\n', ncc_AT);
fprintf('NCC(A,C)      : %.4f\n', ncc_AC);
fprintf('PSNR(A,C)     : %.2f dB\n', psnr_AC);
fprintf('SSIM(A,C)     : %.4f\n', ssim_AC);
fprintf('PSNR(A,T)     : %.2f dB\n', psnr_AT);
fprintf('SSIM(A,T)     : %.4f\n', ssim_AT);
fprintf('Pa/Pr         : %.4e (%.2f dB)\n', PaPr_final, PaPr_final_dB);

%% ------------------------------------------------------------------------
% Visualization: clean, target, attacked, and difference
% ------------------------------------------------------------------------
figure();

subplot(2,2,1);
imagesc(clean_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title(sprintf('Clean %s output image', upper(params.sar_algo)));

subplot(2,2,2);
imagesc(target_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title('Target image (desired attacked)');

subplot(2,2,3);
imagesc(adv_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title('Adversarial image (attacked)');

subplot(2,2,4);
imagesc(adv_img - clean_img);
colormap gray; colorbar;
set(gca, 'YDir', 'normal');
axis image off;
title('Diff (A-C)');

%% ------------------------------------------------------------------------
% Optional: save images for paper / slides / Visio use
% ------------------------------------------------------------------------
visio_fmt = 'png';  % options: 'svg', 'emf', 'pdf', 'png'

if save_final_images
    victim_obj = char(rawData(rawData_select));

    if strcmpi(target_mode, 'noise')
        target_suffix = 'tgt_noise';
        target_obj_for_filename = 'noise';
    else
        target_obj = char(rawData(target_raw_select));
        target_suffix = sprintf('tgt_%s', target_obj);
        target_obj_for_filename = target_obj;
    end

    algo_lower = lower(params.sar_algo);
    algo_obj_folder = sprintf('%s_%s', algo_lower, victim_obj);

    if use_pa_pr_projection
        constraint_folder = 'power_constraint';
    else
        constraint_folder = 'no_power_constraint';
    end

    out_dir = fullfile('figures', constraint_folder, algo_obj_folder);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    img_sets = {
        'clean',  clean_img,           sprintf('%s_%s', victim_obj, target_suffix);
        'target', target_img,          target_obj_for_filename;
        'adv',    adv_img,             sprintf('%s_%s', victim_obj, target_suffix);
        'diff',   adv_img - clean_img, sprintf('%s_%s', victim_obj, target_suffix)
    };

    for i = 1:size(img_sets, 1)
        prefix    = img_sets{i,1};
        img_data  = img_sets{i,2};
        name_part = img_sets{i,3};

        out_file = fullfile(out_dir, sprintf('%s_%s_%s.%s', ...
            prefix, algo_lower, name_part, lower(visio_fmt)));

        f = figure('Visible', 'off', 'Color', 'w');
        imagesc(img_data);
        colormap gray;
        colorbar;
        set(gca, 'YDir', 'normal');
        axis image off;

        switch lower(visio_fmt)
            case 'svg'
                try
                    exportgraphics(f, out_file, 'ContentType', 'vector');
                catch
                    print(f, out_file, '-dsvg');
                end

            case 'emf'
                if ispc
                    try
                        exportgraphics(f, out_file, 'ContentType', 'vector');
                    catch
                        print(f, out_file, '-dmeta');
                    end
                else
                    warning('EMF export is mainly supported on Windows. Saving PDF instead.');
                    out_file = fullfile(out_dir, sprintf('%s_%s_%s.pdf', ...
                        prefix, algo_lower, name_part));
                    exportgraphics(f, out_file, 'ContentType', 'vector');
                end

            case 'pdf'
                exportgraphics(f, out_file, 'ContentType', 'vector');

            case 'png'
                exportgraphics(f, out_file, 'Resolution', 300);

            otherwise
                close(f);
                error('Unsupported visio_fmt: %s. Use ''svg'', ''emf'', ''pdf'', or ''png''.', visio_fmt);
        end

        close(f);
    end

    fprintf('\nIndividual images saved to "%s"\n', out_dir);
end

%% ========================================================================
% FUNCTIONS
% ========================================================================

%% ------------------------------------------------------------------------
% Loss and gradient
% ------------------------------------------------------------------------
function [loss, gradRe, gradIm, atkImage] = loss_and_grad(X_v, D, A_re, A_im, params, lambda_L2)
% LOSS_AND_GRAD
% Computes image-domain loss and gradients with respect to A_re and A_im.
%
% Supported algorithms:
%   params.sar_algo = 'MFA' | 'RMA' | 'BPA' | 'LIA'

    if ~isa(X_v, 'dlarray'), X_v = dlarray(X_v); end
    if ~isa(D,   'dlarray'), D   = dlarray(D);   end
    if nargin < 6 || isempty(lambda_L2), lambda_L2 = 0; end

    A = A_re + 1j * A_im;   % Np x 1
    Y = X_v + D .* A.';     % Nsamp x Np

    Y_cube = reshape(Y, params.Nsamp, params.M, params.N);

    algo = upper(params.sar_algo);
    switch algo
        case 'MFA'
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :));
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end
            [~, ~, atkImage, ~] = dlMFA(sarData, params);

        case 'RMA'
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :));
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end
            [~, ~, atkImage, ~] = dlRMA(sarData, params);

        case 'BPA'
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :));
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end
            if ~isfield(params, 'H_bpa')
                error('params.H_bpa is required for BPA in loss_and_grad.');
            end
            [~, ~, atkImage, ~] = dlBPA(sarData, params, params.H_bpa);

        case 'LIA'
            rawDataFFT = fft(Y_cube, params.nFFTtime);
            sarData    = squeeze(rawDataFFT(params.k0_range_bin + 1, :, :));
            for ii = 2:2:size(sarData, 1)
                sarData(ii, :) = fliplr(sarData(ii, :));
            end
            if ~isfield(params, 'H_bpa')
                error('params.H_bpa is required for LIA in loss_and_grad.');
            end
            if ~isfield(params, 'py')
                error('params.py is required for LIA in loss_and_grad.');
            end
            % Use BPA in the gradient path for speed
            [~, ~, atkImage, ~] = dlBPA(sarData, params, params.H_bpa);

        otherwise
            error('Unknown params.sar_algo = %s', params.sar_algo);
    end

    if strcmpi(params.sar_algo, 'LIA')
        atkImage = atkImage / params.global_scale_bpa;
    else
        atkImage = atkImage / params.global_scale;
    end

    loss_im = mean((atkImage - params.target_img).^2, 'all');
    reg     = lambda_L2 * mean(abs(A).^2, 'all');
    loss    = loss_im + reg;

    [gradRe, gradIm] = dlgradient(loss, A_re, A_im);
end

%% ------------------------------------------------------------------------
% LIA (Li & Chen iterative imaging)
% ------------------------------------------------------------------------
function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlLIA(sarData, params, H_bpa)
% DLLIA
% Lightweight iterative imaging algorithm that reuses the BPA propagation
% matrix H_bpa.
%
% Inputs:
%   sarData : M x N complex gated slice
%   params  : struct with imaging settings and index subset params.py
%   H_bpa   : (M*N) x (A*B) BPA propagation matrix
%
% Outputs:
%   xRangeT, yRangeT   : spatial axes (mm)
%   trueImage_abs      : magnitude image
%   trueImage_complx   : complex image

    if ~isa(sarData, 'dlarray')
        sarData = dlarray(sarData);
    end
    if ~isa(H_bpa, 'dlarray')
        H_bpa = dlarray(H_bpa);
    end

    M  = params.M;
    N  = params.N;
    A  = params.A_bpa;
    B  = params.B_bpa;
    py = params.py;
    bbox = params.bbox;
    dx   = params.dx;
    dy   = params.dy;

    % Vectorize measurements and keep only the selected subset
    rd_full = reshape(sarData, [], 1);
    rd      = rd_full(py);

    % Subsampled propagation matrix
    Hp = H_bpa(py, :);
    BA = A * B;

    % LIA update
    di = 0.01;
    G  = di * (Hp' * Hp);
    xd = di * (Hp' * rd);

    for j = 1:BA
        Gj    = G(:, j);
        denom = 1 + G(j, j);
        temp  = Gj / denom;

        xd = xd - temp * xd(j);
        G  = G  - temp * G(j, :);
    end

    % Extract diagonal as column vector (dlarray-safe)
    BA     = size(G, 1);
    diagG  = G(1:BA+1:BA*BA);
    diagG  = reshape(diagG, [BA, 1]);

    xd = xd ./ diagG;

    % Reshape to B x A and flip horizontally
    xdi = fliplr(reshape(xd, B, A));

    trueImage_complx = xdi;
    trueImage_abs    = dlarray(abs(trueImage_complx), 'SS');

    xRangeT = bbox(1) + (0:size(trueImage_abs, 2) - 1) * dx;
    yRangeT = bbox(3) + (0:size(trueImage_abs, 1) - 1) * dy;
end

%% ------------------------------------------------------------------------
% BPA
% ------------------------------------------------------------------------
function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlBPA(sarData, params, H)
% DLBPA
% Back-Projection Algorithm (BPA) wrapper.

    A = params.A_bpa;
    B = params.B_bpa;

    if ~isa(sarData, 'dlarray')
        sarData = dlarray(sarData);
    end

    % Vectorized measurements
    y = reshape(sarData, [], 1);

    % Standard back-projection
    xd = H' * y;

    % Reshape to image
    xdi = reshape(xd, B, A);

    % Match existing orientation convention
    trueImage_cropped = fliplr(xdi);
    trueImage_complx  = trueImage_cropped;
    trueImage_abs     = dlarray(abs(trueImage_complx), 'SS');

    xRangeT = params.bbox(1) + (0:size(trueImage_abs, 2) - 1) * params.dx;
    yRangeT = params.bbox(3) + (0:size(trueImage_abs, 1) - 1) * params.dy;
end

function H = dlBPA_H_matrix(params)
% DLBPA_H_MATRIX
% Build the BPA propagation matrix H with size:
%   (# measurements) x (# image pixels)

    M = params.M;
    N = params.N;
    A = params.A_bpa;
    B = params.B_bpa;

    c0   = physconst('lightspeed');
    F0   = params.F0;
    z0_m = params.z0 * 1e-3;
    dxm  = params.dx * 1e-3;
    dym  = params.dy * 1e-3;
    bbox_m = params.bbox * 1e-3;

    k   = 2 * pi * F0 / c0;
    cst = 1i * 2 * k;
    z2  = z0_m^2;

    % Image pixel coordinates
    wh1 = linspace(bbox_m(1), bbox_m(2), A);
    wh2 = linspace(bbox_m(3), bbox_m(4), B);

    NM = M * N;
    BA = A * B;

    H_val = complex(zeros(NM, BA));
    fprintf('    Building H matrix (%d x %d)...', NM, BA);
    tic;

    % Keep indexing consistent with y = reshape(sarData, [], 1)
    Ny = params.M;
    Nx = params.N;

    NM = Ny * Nx;
    BA = A * B;

    H_val = complex(zeros(NM, BA));
    fprintf('    Building H matrix (%d x %d)...', NM, BA);
    tic;

    for i = 1:NM
        iy = mod(i - 1, Ny);
        ix = (i - 1 - iy) / Ny;

        sx_i = (ix + 0.5 - Nx / 2) * dxm;
        sy_i = (iy + 0.5 - Ny / 2) * dym;

        for j = 1:BA
            jy = mod(j - 1, B);
            jx = (j - 1 - jy) / B;

            px = wh1(jx + 1);
            py = wh2(jy + 1);

            dist2 = (sx_i - px)^2 + (sy_i - py)^2 + z2;
            H_val(i, j) = exp(cst * sqrt(dist2));
        end
    end

    fprintf([' ' num2str(toc, '%.3f') ' sec\n']);
    H = dlarray(H_val);
end

%% ------------------------------------------------------------------------
% MFA
% ------------------------------------------------------------------------
function matchedFilter = refMF(params)
% REFMF
% Build the 2D matched filter used by MFA.

    c = physconst('lightspeed');
    x = params.dx * (-(params.nFFTspace-1)/2 : (params.nFFTspace-1)/2) * 1e-3;
    y = (params.dy * (-(params.nFFTspace-1)/2 : (params.nFFTspace-1)/2) * 1e-3).';
    z0_m = params.z0 * 1e-3;
    k = 2 * pi * params.F0 / c;

    matchedFilter = exp(-1i * 2 * k * sqrt(bsxfun(@plus, x.^2, y.^2) + z0_m^2));
end

function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlMFA(sarData, params)
% DLMFA
% 2D matched-filter algorithm for SAR image reconstruction.

    matchedFilter = refMF(params);
    if isa(sarData, 'dlarray') && ~isa(matchedFilter, 'dlarray')
        matchedFilter = dlarray(matchedFilter);
    end

    [yPointM, xPointM] = size(sarData);
    [yPointF, xPointF] = size(matchedFilter);

    % Zero-pad sarData to match the matched filter size
    if (xPointF > xPointM)
        pad_x_pre  = floor((xPointF - xPointM) / 2);
        pad_x_post = ceil((xPointF - xPointM) / 2);
        sarData = cat(2, zeros(yPointM, pad_x_pre, 'like', sarData), ...
                         sarData, ...
                         zeros(yPointM, pad_x_post, 'like', sarData));
        xPointM = xPointF;
    end

    if (yPointF > yPointM)
        pad_y_pre  = floor((yPointF - yPointM) / 2);
        pad_y_post = ceil((yPointF - yPointM) / 2);
        sarData = cat(1, zeros(pad_y_pre, xPointM, 'like', sarData), ...
                         sarData, ...
                         zeros(pad_y_post, xPointM, 'like', sarData));
        yPointM = yPointF;
    end

    % Frequency-domain matched filtering
    sarDataFFT       = fft(fft(sarData, [], 2), [], 1);
    matchedFilterFFT = fft(fft(matchedFilter, [], 2), [], 1);
    trueImage_shifted = ifft(ifft(sarDataFFT .* matchedFilterFFT, [], 2), [], 1);

    % Center and crop
    trueImage = fftshift(trueImage_shifted);
    [J, I] = size(trueImage);

    xij = round(params.bbox(1:2) / params.dx - 0.5 + I / 2);
    ykl = round(params.bbox(3:4) / params.dy - 0.5 + J / 2);

    trueImage_cropped = trueImage(ykl(1):ykl(2), xij(1):xij(2));
    trueImage_cropped = fliplr(trueImage_cropped);
    trueImage_complx  = trueImage_cropped;
    trueImage_abs     = dlarray(abs(trueImage_cropped), 'SS');

    xRangeT = params.bbox(1) + (0:size(trueImage_abs, 2) - 1) * params.dx;
    yRangeT = params.bbox(3) + (0:size(trueImage_abs, 1) - 1) * params.dy;
end

%% ------------------------------------------------------------------------
% RMA
% ------------------------------------------------------------------------
function [xRangeT, yRangeT, trueImage_abs, trueImage_complx] = dlRMA(sarData, params)
% DLRMA
% 2D range-migration algorithm using dlarray-compatible operations.

    nFFTspace = params.nFFTspace;
    z0_mm     = params.z0;
    dx        = params.dx;
    dy        = params.dy;
    bbox      = params.bbox;
    F0        = params.F0;

    isDlArray = isa(sarData, 'dlarray');
    if ~isDlArray
        sarData = dlarray(sarData);
    end

    % Spatial frequency setup
    c = physconst('lightspeed');
    k = 2 * pi * F0 / c;

    wSx = 2 * pi / (dx * 1e-3);
    kX  = linspace(-(wSx / 2), (wSx / 2), nFFTspace);

    wSy = 2 * pi / (dy * 1e-3);
    kY  = (linspace(-(wSy / 2), (wSy / 2), nFFTspace)).';

    K = sqrt((2 * k).^2 - bsxfun(@plus, kX.^2, kY.^2));

    if ~isDlArray
        K = dlarray(K);
    end

    % Range migration phase term
    phaseFactor0 = exp(-1i * z0_mm * K);

    % Remove evanescent terms
    K_mag_sq = bsxfun(@plus, kX.^2, kY.^2);
    evanescent_mask = K_mag_sq > (2 * k)^2;

    phaseFactor0(evanescent_mask) = 0;
    phaseFactor0 = dlarray(phaseFactor0);

    phaseFactor = K .* phaseFactor0;
    phaseFactor = fftshift(fftshift(phaseFactor, 1), 2);

    % Match dimensions with zero padding
    [yPointM, xPointM] = size(sarData);
    [yPointF, xPointF] = size(phaseFactor);

    if (xPointF > xPointM)
        pad_x_pre  = floor((xPointF - xPointM) / 2);
        pad_x_post = ceil((xPointF - xPointM) / 2);
        sarData = cat(2, zeros(yPointM, pad_x_pre, 'like', sarData), ...
                         sarData, ...
                         zeros(yPointM, pad_x_post, 'like', sarData));
    elseif (xPointM > xPointF)
        pad_x_pre  = floor((xPointM - xPointF) / 2);
        pad_x_post = ceil((xPointM - xPointF) / 2);
        phaseFactor = cat(2, zeros(yPointF, pad_x_pre, 'like', phaseFactor), ...
                             phaseFactor, ...
                             zeros(yPointF, pad_x_post, 'like', phaseFactor));
    end

    if (yPointF > yPointM)
        pad_y_pre  = floor((yPointF - yPointM) / 2);
        pad_y_post = ceil((yPointF - yPointM) / 2);
        sarData = cat(1, zeros(pad_y_pre, size(sarData, 2), 'like', sarData), ...
                         sarData, ...
                         zeros(pad_y_post, size(sarData, 2), 'like', sarData));
    elseif (yPointM > yPointF)
        pad_y_pre  = floor((yPointM - yPointF) / 2);
        pad_y_post = ceil((yPointM - yPointF) / 2);
        phaseFactor = cat(1, zeros(pad_y_pre, size(phaseFactor, 2), 'like', phaseFactor), ...
                             phaseFactor, ...
                             zeros(pad_y_post, size(phaseFactor, 2), 'like', phaseFactor));
    end

    % Image formation
    sarDataFFT = fft(fft(sarData, [], 2), [], 1);
    trueImage  = ifft(ifft(sarDataFFT .* phaseFactor, [], 2), [], 1);

    % Crop
    [J, I] = size(trueImage);

    xij = round(bbox(1:2) / dx - 0.5 + I / 2);
    ykl = round(bbox(3:4) / dy - 0.5 + J / 2);

    trueImage_cropped = trueImage(ykl(1):ykl(2), xij(1):xij(2));
    trueImage_cropped = fliplr(trueImage_cropped);
    trueImage_complx  = trueImage_cropped;
    trueImage_abs     = abs(trueImage_cropped);

    xRangeT = bbox(1) + (0:size(trueImage_abs, 2) - 1) * dx;
    yRangeT = bbox(3) + (0:size(trueImage_abs, 1) - 1) * dy;
end