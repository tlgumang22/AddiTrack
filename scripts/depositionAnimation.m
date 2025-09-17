% ==== Deposition 3D Animation ====
% Reads Time, Length, Height from Excel
% Plots extrusion with Length = X, Height = Z (vertical), Width = constant

clc; clear; close all;

% ---- Load Excel data ----
data = readmatrix('simulated_series_V7_I14.5_F48_T100.xlsx');

Time   = data(:,1);   % Time (s)
Height = data(:,2);   % Height (Z-axis, vertical)
Length = data(:,3);   % Length (X-axis)

% ---- Assume constant width ----
Width = ones(size(Length)) * 2;   % mm (adjust as needed)

% Symmetric about Y=0
Y1 = -Width/2;
Y2 =  Width/2;

% ---- Axis limits ----
Xrange = [min(Length) max(Length)];
Yrange = [min(Y1) max(Y2)];
Zrange = [0 max(Height)];

% ---- Animation settings ----
points_per_sec   = 40;        % speed of plotting
update_interval  = 0.2;       % seconds per update
points_per_step  = round(points_per_sec * update_interval);
nSteps = ceil(length(Length)/points_per_step);

% ---- Setup figure with multiple views ----
figure('Color','w');
tiledlayout(2,2,"TileSpacing","compact","Padding","compact");

% Isometric view (top row)
ax1 = nexttile([1 2]);  
xlabel(ax1,'Length (X, mm)');
ylabel(ax1,'Width (Y, mm)');
zlabel(ax1,'Height (Z, mm)');
title(ax1,'Isometric View (45°)','FontWeight','bold');
grid(ax1,"on");
xlim(ax1,Xrange); ylim(ax1,Yrange); zlim(ax1,Zrange);
view(ax1,45,30);

% Side view (bottom-left)
ax2 = nexttile;  
xlabel(ax2,'Length (X, mm)');
ylabel(ax2,'Width (Y, mm)');
zlabel(ax2,'Height (Z, mm)');
title(ax2,'Side View','FontWeight','bold');
grid(ax2,"on");
xlim(ax2,Xrange); ylim(ax2,Yrange); zlim(ax2,Zrange);
view(ax2,0,0);   % looks along +X

% Top view (bottom-right)
ax3 = nexttile;  
xlabel(ax3,'Width (Y, mm)');
ylabel(ax3,'Length (X, mm)');
title(ax3,'Top View','FontWeight','bold');
grid(ax3,"on");
xlim(ax3,Yrange); ylim(ax3,Xrange); zlim(ax3,Zrange);
view(ax3,90,90);

% Store axes for easy looping
axs = {ax1, ax2, ax3};

% ---- Animate deposition ----
for k = 1:nSteps
    idx = 1:min(k*points_per_step, length(Length));

    for v = 1:3
        ax = axs{v};
        cla(ax);
        hold(ax,'on');

        % Use Length as color gradient
        cData = repmat(Length(idx)',2,1);

        % Left face
        surf(ax,[Length(idx)'; Length(idx)'], [Y1(idx)'; Y1(idx)'], ...
             [zeros(size(Height(idx)))'; Height(idx)'], ...
             cData, 'EdgeColor','none','FaceAlpha',0.95); 

        % Right face
        surf(ax,[Length(idx)'; Length(idx)'], [Y2(idx)'; Y2(idx)'], ...
             [zeros(size(Height(idx)))'; Height(idx)'], ...
             cData, 'EdgeColor','none','FaceAlpha',0.95); 

        % Top face
        surf(ax,[Length(idx)'; Length(idx)'], [Y1(idx)'; Y2(idx)'], ...
             [Height(idx)'; Height(idx)'], ...
             cData, 'EdgeColor','none','FaceAlpha',0.95);

        % Bottom face
        surf(ax,[Length(idx)'; Length(idx)'], [Y1(idx)'; Y2(idx)'], ...
             zeros(2,length(idx)), ...
             cData, 'EdgeColor','none','FaceAlpha',0.8);

        % End caps
        Lf = Length(idx(end));
        Hf = Height(idx(end));
        surf(ax,[Lf Lf; Lf Lf], [Y1(idx(end)) Y2(idx(end)); Y1(idx(end)) Y2(idx(end))], ...
                 [0 0; Hf Hf], ...
             Lf*ones(2,2), 'EdgeColor','none','FaceAlpha',0.95);

        Lb = Length(idx(1));
        Hb = Height(idx(1));
        surf(ax,[Lb Lb; Lb Lb], [Y1(idx(1)) Y2(idx(1)); Y1(idx(1)) Y2(idx(1))], ...
                 [0 0; Hb Hb], ...
             Lb*ones(2,2), 'EdgeColor','none','FaceAlpha',0.95);

        % Appearance
        colormap(ax,turbo);  
        shading(ax,"interp");   
        camlight(ax,"headlight"); 
        lighting(ax,"gouraud");
    end

    drawnow;
    pause(update_interval);
end
