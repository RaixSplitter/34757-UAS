close all; clc

model = 'quadrotor_model';

m   = 0.5;          % kg
L   = 0.225;        % m
k = 0.01;         % N·s²/rad²   
b   = 0.001;        % N·m·s²/rad²
D   = diag([.01 .01 .01]);       % N·s/m
Ixx = 3e-6;  Iyy = 3e-6;  Izz = 1e-5;
I   = diag([Ixx Iyy Izz]);
g_i = [0 0 -9.81]';

p0  = [0 0 0]';  v0 = [0 0 0]';
eul0= [0 0 0]';  w0 = [0 0 0]';

testcase1 = [0;     0;     0;     0];
testcase2 = [1e4;   0;     1e4;   0];
testcase3 = [0;     1e4;   0;     1e4];
cases = {"All-off", testcase1;
         "Roll"   , testcase2;
         "Pitch"  , testcase3};

for i = 1:3
    omega = cases{i,2};
    simout = sim(model,'StopTime','3');

    t  = simout.tout;
    p  = squeeze(simout.p.Data);
    ang= squeeze(simout.euler.Data);

    % 2-D plots
    figure(1), subplot(3,1,i)
    plot(t,p), grid on, title(['p(t) – ' cases{i,1}]), legend x y z
    figure(2), subplot(3,1,i)
    plot(t,ang*180/pi), grid on
    title(['\Theta(t) – ' cases{i,1}]), legend \phi \theta \psi
end