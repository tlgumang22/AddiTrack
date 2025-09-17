% inverse_control_and_animate.m
% 1) Trains inverse models (Height,Time,Length -> V,I,F) if needed
% 2) Runs your 3D deposition animation (uses interpolated_deposition.csv)
% 3) Every checkpointInterval (default 10 s) predicts V,I,F to maintain targetHeight
% 4) Overlays predicted params on plots and saves them to CSV

clearvars; close all; clc;

%% ----------------- USER CONFIG -----------------
trainingCSV      = 'training_data.csv';            % must contain Length, Time, Height, Voltage, Current, FeedRate
interpCSV        = 'interpolated_deposition.csv'; % your existing CSV with Length,Height,Width
modelFile        = 'inverse_models.mat';
outputPredCSV    = 'predicted_ivf_for_constant_height.csv';

targetHeight     = [];     % if empty -> default to mean of Height in interpolated CSV
checkpointInterval = 10;   % seconds (predict/update IVF each 10s)
points_per_sec   = 40;     % used to build time vector if Time not present
update_interval  = 0.2;    % seconds per animation update
numTrees         = 200;    % for fitrensemble / TreeBagger
%% ------------------------------------------------

%% 0) Read interpolation CSV (the geometry used for animation)
A = readmatrix(interpCSV);
Length = A(:,1);
Height = A(:,2);
Width  = A(:,3);

N = numel(Length);

% symmetric width
Y1 = -Width/2;
Y2 =  Width/2;

% default targetHeight if not provided
if isempty(targetHeight)
    targetHeight = mean(Height); 
    fprintf('targetHeight empty -> using mean(Height)=%.3f mm\n', targetHeight);
end

% Build/estimate time vector for the interpolated deposition points
% If you have a separate time column in the CSV, replace this.
time_per_point = 1/points_per_sec;
timeVec = (1:N)' * time_per_point;   % seconds

%% 1) Train inverse models if missing
if ~exist(modelFile,'file')
    fprintf('Model file not found. Attempting to train inverse models from %s\n', trainingCSV);
    if ~exist(trainingCSV,'file')
        error('Training CSV not found. Provide %s with columns: Length,Time,Height,Voltage,Current,FeedRate', trainingCSV);
    end
    T = readtable(trainingCSV);

    req = {'Length','Time','Height','Voltage','Current','FeedRate'};
    if ~all(ismember(req, T.Properties.VariableNames))
        error('training CSV must contain columns: %s', strjoin(req,','));
    end

    % Build features and targets
    X = [T.Height, T.Time, T.Length];   % [Height, Time, Length]
    yV = T.Voltage;
    yI = T.Current;
    yF = T.FeedRate;

    % Remove rows with NaNs
    valid = all(~isnan([X yV yI yF]),2);
    X = X(valid,:);
    yV = yV(valid);
    yI = yI(valid);
    yF = yF(valid);

    % Normalize inputs (store mu/sigma)
    muX = mean(X,1);
    sigmaX = std(X,[],1);
    sigmaX(sigmaX==0) = 1;

    Xn = (X - muX) ./ sigmaX;

    % Train three regressors (one per output). Using bagged trees as robust baseline.
    try
        rng(0); % reproducible
        mdlV = fitrensemble(Xn, yV, 'Method','Bag', 'NumLearningCycles', numTrees);
        mdlI = fitrensemble(Xn, yI, 'Method','Bag', 'NumLearningCycles', numTrees);
        mdlF = fitrensemble(Xn, yF, 'Method','Bag', 'NumLearningCycles', numTrees);
    catch ME
        warning('fitrensemble failed (maybe toolbox missing). Trying TreeBagger fallback.');
        mdlV = TreeBagger(numTrees, Xn, yV, 'Method','regression');
        mdlI = TreeBagger(numTrees, Xn, yI, 'Method','regression');
        mdlF = TreeBagger(numTrees, Xn, yF, 'Method','regression');
    end

    % Save training ranges for clipping later
    trainStats.Vmin = min(yV); trainStats.Vmax = max(yV);
    trainStats.Imin = min(yI); trainStats.Imax = max(yI);
    trainStats.Fmin = min(yF); trainStats.Fmax = max(yF);

    save(modelFile, 'mdlV','mdlI','mdlF','muX','sigmaX','trainStats');
    fprintf('Training complete. Models saved to %s\n', modelFile);
else
    fprintf('Loading inverse models from %s\n', modelFile);
    load(modelFile, 'mdlV','mdlI','mdlF','muX','sigmaX','trainStats');
    if ~exist('muX','var') || isempty(muX)
        error('Model file missing muX/sigmaX. Retrain with training CSV.');
    end
end

%% 2) Setup animation axes (similar to your code)
Xrange = [min(Length) max(Length)];
Yrange = [min(Y1) max(Y2)];
Zrange = [0 max(Height)*1.1];

points_per_step  = round(points_per_sec * update_interval);
nSteps = ceil(N / points_per_step);

figure('Color','w','Units','normalized','Position',[0.05 0.05 0.9 0.85]);
tiledlayout(2,2,"TileSpacing","compact","Padding","compact");

% Isometric view
ax1 = nexttile([1 2]);
xlabel(ax1,'Length (X)'); ylabel(ax1,'Width (Y)'); zlabel(ax1,'Height (Z)');
title(ax1,'Isometric View (45°)','FontWeight','bold');
grid(ax1,"on"); xlim(ax1,Xrange); ylim(ax1,Yrange); zlim(ax1,Zrange);
view(ax1,45,30);

% Side view
ax2 = nexttile;
xlabel(ax2,'Length (X)'); ylabel(ax2,'Width (Y)'); zlabel(ax2,'Height (Z)');
title(ax2,'Side View','FontWeight','bold');
grid(ax2,"on"); xlim(ax2,Xrange); ylim(ax2,Yrange); zlim(ax2,Zrange);
view(ax2,0,0);

% Top view
ax3 = nexttile;
xlabel(ax3,'Width (Y)'); ylabel(ax3,'Length (X)');
title(ax3,'Top View','FontWeight','bold');
grid(ax3,"on"); xlim(ax3,Xrange); ylim(ax3,Yrange); % zlim not used for top
view(ax3,90,90);

axs = {ax1, ax2, ax3};

% UI text box at bottom to show latest predicted parameters
htxt = uicontrol('Style','text','Units','normalized','Position',[0.02 0.01 0.96 0.06],...
    'BackgroundColor','w','FontSize',11,'HorizontalAlignment','left','String','Waiting for predictions...');

% Prepare results container
results = []; % columns: time, idx, length, V, I, F

nextCheckpoint = 0;

%% 3) Animation + Inverse control loop
fprintf('Starting animation + inverse-control loop. Checkpoints every %d s.\n', checkpointInterval);
for k = 1:nSteps
    idx = 1 : min(k * points_per_step, N);

    % Draw surfaces on each axis
    for v = 1:3
        ax = axs{v};
        cla(ax);
        hold(ax,'on');

        cData = repmat(Length(idx)',2,1);

        % left face
        surf(ax, [Length(idx)'; Length(idx)'], [Y1(idx)'; Y1(idx)'], ...
                 [zeros(size(Height(idx)))'; Height(idx)'], cData, 'EdgeColor','none','FaceAlpha',0.95);

        % right face
        surf(ax, [Length(idx)'; Length(idx)'], [Y2(idx)'; Y2(idx)'], ...
                 [zeros(size(Height(idx)))'; Height(idx)'], cData, 'EdgeColor','none','FaceAlpha',0.95);

        % top face
        surf(ax, [Length(idx)'; Length(idx)'], [Y1(idx)'; Y2(idx)'], ...
                 [Height(idx)'; Height(idx)'], cData, 'EdgeColor','none','FaceAlpha',0.95);

        % bottom face
        surf(ax, [Length(idx)'; Length(idx)'], [Y1(idx)'; Y2(idx)'], ...
                 zeros(2,length(idx)), cData, 'EdgeColor','none','FaceAlpha',0.8);

        % end caps (front & back)
        Lf = Length(idx(end)); Hf = Height(idx(end));
        surf(ax, [Lf Lf; Lf Lf], [Y1(idx(end)) Y2(idx(end)); Y1(idx(end)) Y2(idx(end))], ...
                 [0 0; Hf Hf], Lf*ones(2,2), 'EdgeColor','none','FaceAlpha',0.95);

        Lb = Length(idx(1)); Hb = Height(idx(1));
        surf(ax, [Lb Lb; Lb Lb], [Y1(idx(1)) Y2(idx(1)); Y1(idx(1)) Y2(idx(1))], ...
                 [0 0; Hb Hb], Lb*ones(2,2), 'EdgeColor','none','FaceAlpha',0.95);

        colormap(ax, turbo);
        shading(ax, "interp");
        camlight(ax,"headlight");
        lighting(ax,"gouraud");
    end

    drawnow;

    % --- Check if we crossed a checkpoint (every checkpointInterval seconds) ---
    current_time = timeVec(idx(end));
    if current_time >= nextCheckpoint
        % Build prediction input: [Height_target, Time, Length_at_current]
        Xpred = [targetHeight, current_time, Length(idx(end))];
        Xpredn = (Xpred - muX) ./ sigmaX;  % normalize

        % Ensure shape (row)
        if isrow(Xpredn), Xpredn_row = Xpredn; else Xpredn_row = Xpredn'; end

        % Predict using models (handle both fitrensemble and TreeBagger)
        try
            Vpred = predict(mdlV, Xpredn_row);
            Ipred = predict(mdlI, Xpredn_row);
            Fpred = predict(mdlF, Xpredn_row);
        catch
            % TreeBagger fallback predict
            Vpred = predict(mdlV, Xpredn_row);
            Ipred = predict(mdlI, Xpredn_row);
            Fpred = predict(mdlF, Xpredn_row);
            Vpred = Vpred(1); Ipred = Ipred(1); Fpred = Fpred(1);
        end

        % Clip predictions to training ranges if available
        if exist('trainStats','var')
            Vpred = min(max(Vpred, trainStats.Vmin), trainStats.Vmax);
            Ipred = min(max(Ipred, trainStats.Imin), trainStats.Imax);
            Fpred = min(max(Fpred, trainStats.Fmin), trainStats.Fmax);
        end

        % Store results
        results = [results; current_time, idx(end), Length(idx(end)), Vpred, Ipred, Fpred];

        % Update text overlay (bottom UI)
        sstr = sprintf('Checkpoint @ t=%.1f s | idx=%d | x=%.3f mm  -->  V=%.3f V  |  I=%.3f A  |  F=%.3f mm/min', ...
                       current_time, idx(end), Length(idx(end)), Vpred, Ipred, Fpred);
        set(htxt, 'String', sstr);

        % Also place text in the isometric axes near the current front
        ax = ax1;
        xpos = ax.XLim(1) + 0.02*diff(ax.XLim);
        ypos = ax.YLim(2) - 0.02*diff(ax.YLim);
        zpos = ax.ZLim(2) - 0.02*diff(ax.ZLim);
        % remove previous small label by clearing all texts (simple approach)
        delete(findall(ax, 'Type','text','Tag','ivfText'));
        text(ax, xpos, ypos, zpos, sprintf('t=%.0f s\nV=%.2f V\nI=%.2f A\nF=%.1f', current_time, Vpred, Ipred, Fpred), ...
             'BackgroundColor','w','EdgeColor','k','FontSize',9,'Tag','ivfText','Interpreter','none');

        % move to next checkpoint time
        nextCheckpoint = nextCheckpoint + checkpointInterval;
    end

    pause(update_interval);
end

%% 4) Save predicted results table
if ~isempty(results)
    Tpred = array2table(results, 'VariableNames', {'Time_s','Idx','Length_mm','Voltage_V','Current_A','FeedRate_mm_per_min'});
    writetable(Tpred, outputPredCSV);
    fprintf('Predicted IVF table saved to %s\n', outputPredCSV);
else
    fprintf('No predictions were generated (maybe checkpointInterval too large?).\n');
end

fprintf('Done.\n');