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

