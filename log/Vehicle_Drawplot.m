%% Vehicle_vanilla_iLQR State
figure(1);
hold on
axis equal
plot(trajectory(:,1), trajectory(:,2), '--')
grid on

%% Vehicle_dd_iLQR State
figure(1);
hold on
axis equal
plot(optimal_trajectory(:,1), optimal_trajectory(:,2))
grid on


%% Vehicle_vanilla_iLQR Input
figure(2);
subplot(2,1,1)
hold on
plot(trajectory(:,5), '--')
grid on
subplot(2,1,2)
hold on
plot(trajectory(:,6), '--')
grid on
%% Vehicle_dd_iLQR Input
figure(2);
subplot(2,1,1)
hold on
plot(optimal_trajectory(:,5))
grid on
subplot(2,1,2)
hold on
plot(optimal_trajectory(:,6))
grid on
%% Vehicle_dd_iLQR (Compare noisy and optimal)
figure(3);
hold on
axis equal
plot(optimal_trajectory(:,1), optimal_trajectory(:,2))
plot(trajectroy_noisy(:,1), trajectroy_noisy(:,2), '--')
grid on

figure(4);
subplot(2,1,1)
hold on
plot(optimal_trajectory(:,5))
plot(trajectroy_noisy(:,5), '--')
grid on
subplot(2,1,2)
hold on
plot(optimal_trajectory(:,6))
plot(trajectroy_noisy(:,6), '--')
grid on

%% Vehicle Dynamic
figure(5);
axis equal
box on
xlim([-5,80])
ylim([-10 14])
xticks([0 10 20 30 40 50])
hold on
grid minor

DATA_LEN = 100;
plot(result(:,1),result(:,2), 'LineWidth', 2, 'Color', [0.4940, 0.1840, 0.5560])
x = -5:0.1:80-0.1;
plot(x,linspace(-3,-3,length(x)), 'LineStyle', '--' ,'LineWidth', 2, 'Color', 'black')
for tau=1:1:DATA_LEN
    
    car = patch('XData',[0 0 3 3 0]+result(tau,1),'YData',[0 -1 -1 1 1]+result(tau,2),'FaceColor',[0,0.4470,0.7410]);
    rotate(car,[0 0 1],rad2deg(result(tau,3)),[result(tau,1), result(tau,2), 0]);
    pause(0.1)
    
end

