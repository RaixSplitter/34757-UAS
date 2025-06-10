close all; clc

model = 'quadrotor_model_2';

m   = 0.5;          % kg
L   = 0.225;        % m
k = 0.01;         % N·s²/rad²   
b   = 0.001;        % N·m·s²/rad²
D   = diag([.01 .01 .01]);       % N·s/m
Ixx = 3e-6;  Iyy = 3e-6;  Izz = 1e-5;
I   = diag([Ixx Iyy Izz]);
g_i = [0 0 -9.81]';

p0  = [0 0 0]';  v0 = [0 0 0]';
q0= [1 0 0 0]';  w0 = [0 0 0]';

%testcase1 = [11.11;     11.11;     11.11;     11.11];
testcase1 = [0, 0, 0, 0];
testcase2 = [1e4;   0;     1e4;   0];
testcase3 = [0;     1e4;   0;     1e4];
cases = {"All-off", testcase1;
         "Roll"   , testcase2;
         "Pitch"  , testcase3};

ios(1) = linio(strcat(model,'/Gain1'),1,'openinput');
ios(2) = linio(strcat(model, '/Gain'),1,'openoutput');
omega = testcase2;
linsys=linearize(model,ios);

%{
for i = 1:3
    omega = cases{i,2};
    simout = sim(model,'StopTime','3');

    t  = simout.tout;
    p  = squeeze(simout.p.Data);
    q_sim = squeeze(simout.q.Data);   % 4 × N
    q_sim = q_sim.';                 % N × 4   ← what quat2eul expects
    
    ang   = quat2eul(q_sim, 'ZYX');  % yaw-pitch-roll, scalar-first
    ang   = ang(:, [3 2 1]);         % reorder to  φ θ ψ
    
    % 2-D plots
    figure(1), subplot(3,1,i)
    plot(t,p), grid on, title(['p(t) – ' cases{i,1}]), legend x y z
    figure(2), subplot(3,1,i)
    plot(t,ang*180/pi), grid on
    title(['\Theta(t) – ' cases{i,1}]), legend \phi \theta \psi
end
%}