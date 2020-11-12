%% Cartpole1_vanilla_iLQR State
figure(1);
grid minor
hold on
plot(trajectory(:,1), '--')
plot(trajectory(:,2), '--')


%% Cartpole1_dd_iLQR State
figure(1);
grid minor
hold on
plot(optimal_trajectory(:,1))
plot(optimal_trajectory(:,2))



%% Cartpole1_vanilla_iLQR Input
figure(2);
grid minor
hold on
plot(trajectory(:,5), '--')


%% Cartpole1_dd_iLQR Input
figure(2);
grid minor
hold on
plot(optimal_trajectory(:,5))

%% Cartpole1_dd_iLQR (Compare noisy and optimal)
figure(3);
hold on
plot(optimal_trajectory(:,1))
plot(optimal_trajectory(:,2))
plot(trajectroy_noisy(:,1), '--')
plot(trajectroy_noisy(:,2), '--')
figure(4);
hold on
plot(optimal_trajectory(:,5))
plot(trajectroy_noisy(:,5), '--')
%% Cart Pole Animation
env = rlPredefinedEnv('CartPole-Discrete');
for i = 1:1:length(optimal_trajectory)
    env.State(1) = optimal_trajectory(i,3);
    env.State(2) = optimal_trajectory(i,4);
    env.State(3) = optimal_trajectory(i,1);
    env.State(4) = optimal_trajectory(i,2);
    plot(env)
end

