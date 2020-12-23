%function DNN_MF

close all
clc
clear all

%%Parameters 
Tf=100;

m=1;
n=3;

Delta_T=0.0001;

TimeSpan=0:Delta_T:Tf;

[mt nt]=size(TimeSpan);

u=zeros(m,nt);

x=zeros(n,nt);

    for j=1:n
        x(j,1)=2*j;
    end

    A_s=0.001*diag([-5,-7,-6]);

    B=[0; 0; 1];    
    
    for i=1:nt

        u(1,i)= 500*sin(0.00004*i)*cos(0.000009*i+1.15);
    
        %for j=1:n
       
           x(:,i+1) = x(:,i) + Delta_T * ( A_s * x(:,i) + B * u(:,i) + [70*rand(1,1);80*rand(1,1);90*rand(1,1)]);
        
        %end

    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%% Neural network structure %%%%%%%%%%%%
    
    

A=diag([-2 -2 -2])*20;              %%User defined

P=diag([60 40 40])*1575.9;           %%User defined or Riccati based solution    

n1=4;                              %%User defined
n2=5;                              %%User defined

xe=9*ones(n,nt);

W1=20*ones(n,n1);
W2=20*ones(n,n2);

W1_S=ones(n,n1,nt);
W2_S=ones(n,n2,nt);

W1_S(:,:,1)=W1(:,:);
W2_S(:,:,1)=W2(:,:);

sigma1=zeros(n1,1,nt);
sigma2=zeros(n2,m,nt);

K1=diag([20,10,20])*0.1;
K2=diag([20,10,20])*0.1;

C=0.1*[1, 1, 1];

for i=1:nt

    %%% Sigma 1
    for j1=1:n1
        sigma1(j1,1)=1/(1+0.22*exp(-0.2*C*x(:,i)))-0.3;    
    end
    
    %%% Sigma 2
    for j21=1:n2
        for j22=1:m
            sigma2(j21,j22)=1/(1+0.2*exp(-0.2*C*x(:,i)))-0.2;    
        end
    end
    
    Delta(:,i)= xe(:,i) - x(:,i);
    
    xe(:,i+1)= xe(:,i) + Delta_T * ( A*xe(:,i) + W1 * sigma1(:,1) + (W2 * sigma2(:,:,i))*u(:,i) ); 

    W1 = W1 - Delta_T *(K1*P*Delta(:,i)*(sigma1(:,1))');

    W2 = W2 - Delta_T * (K2*P*Delta(:,i)*(u(:,i))'*(sigma2(:,:,i))');

    W1_S(:,:,i+1)=W1(:,:);
    W2_S(:,:,i+1)=W2(:,:);
end

figure(1)
hold on
plot(TimeSpan,x(1,1:nt),'r','LineWidth',2)
plot(TimeSpan,xe(1,1:nt),'b','LineWidth',2)
hold off

figure(2)
hold on
plot(TimeSpan,x(2,1:nt),'r','LineWidth',2)
plot(TimeSpan,xe(2,1:nt),'b','LineWidth',2)
hold off

figure(3)
hold on
plot(TimeSpan,x(3,1:nt),'r','LineWidth',2)
plot(TimeSpan,xe(3,1:nt),'b','LineWidth',2)
hold off

%end %%Function


