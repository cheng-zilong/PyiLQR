%% For y
fig=figure(1);
t = linspace(0,2*pi) ;
a = 5 ; b = 2 ;
x = a*cos(t)+15 ;
y = b*sin(t)-1 ;
plot(x,y,'b')
hold on
plot(result(:,1),result(:,2))
axis equal

%% Lane Change
fig5=figure(5);
axis equal
box on
xlim([-5,55])
ylim([-10 14])
xticks([0 10 20 30 40 50])
hold on
grid minor

DATA_LEN = 60;
plot(result(:,1),result(:,2), 'LineWidth', 2, 'Color', [0.4940, 0.1840, 0.5560])
plot(-5:0.1:55-0.1,linspace(6,6,600), 'LineWidth', 2, 'Color', 'black')
plot(-5:0.1:55-0.1,linspace(2,2,600), 'LineWidth', 1, 'Color', 'black')
plot(-5:0.1:55-0.1,linspace(-2,-2,600), 'LineWidth', 2, 'Color', 'black')
plot(-5:0.1:55-0.1,linspace(4,4,600), 'LineStyle', '--' ,'LineWidth', 2, 'Color', 'black')
for tau=1:1:DATA_LEN
    obstacle1_velocity = 6;
    t = linspace(0,2*pi);
    a = 5 ; b = 2.5;
    x = a*cos(t)+obstacle1_velocity*0.1*tau;
    y = b*sin(t)+4;
    obstacle1_car = patch('XData',[0 0 3 3 0]-1.5+obstacle1_velocity*0.1*tau,'YData',[0 -1 -1 1 1]+4,'FaceColor',[0.9290,0.6940,0.1250]);
    obstacle1_circle = plot(x,y,'Color', [0.9290,0.6940,0.1250], 'LineWidth', 2, 'LineStyle', '--');
    
    obstacle2_velocity = 3;
    t = linspace(0,2*pi);
    a = 5 ; b = 2.5;
    x = a*cos(t)+20+obstacle2_velocity*0.1*tau;
    y = b*sin(t);
    obstacle2_car = patch('XData',[0 0 3 3 0]+20-1.5+obstacle2_velocity*0.1*tau,'YData',[0 -1 -1 1 1],'FaceColor',[0.9290,0.6940,0.1250]);
    obstacle2_circle = plot(x,y,'Color', [0.9290,0.6940,0.1250], 'LineWidth', 2, 'LineStyle', '--');

    car = patch('XData',[0 0 3 3 0]+result(tau,1),'YData',[0 -1 -1 1 1]+result(tau,2),'FaceColor',[0,0.4470,0.7410]);
    rotate(car,[0 0 1],rad2deg(result(tau,3)),[result(tau,1), result(tau,2), 0]);
    pause(0.1)
    
    delete(obstacle1_car)
    delete(obstacle1_circle)
    delete(obstacle2_car)
    delete(obstacle2_circle)
end

