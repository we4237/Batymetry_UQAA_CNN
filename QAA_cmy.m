
clc;
clear;
close all;

%%
imgname=uigetfile('*.*', 'Select the image file');
filename=strcat('D:\SHOU\NSG(0717)_1021\',imgname(1:end-4));
hdrname=strcat(imgname(1:end-4), '.hdr');
data=enviread(filename);
img=double(data.z);
[row,col,~]=size(img);


B=img(:,:,1);
G=img(:,:,2);
R=img(:,:,3);
Rrs485=B./200000;
Rrs555=G./200000;
Rrs660=R./200000;

c=zeros(row,col);
ag440=zeros(row,col);
%% QAA_v5算法实现
   %计算rrs

   rrs485=Rrs485./(0.52+1.7*Rrs485);
   rrs555=Rrs555./(0.52+1.7*Rrs555);
   rrs660=Rrs660./(0.52+1.7*Rrs660);
   
   %计算u
    g0=0.08945;
    g1=0.1247; 
    u485=((g0*g0+4.0*g1*rrs485).^(1.0/2.0)-g0)/(2.0*g1);
    u555=((g0*g0+4.0*g1*rrs555).^(1.0/2.0)-g0)/(2.0*g1);
    u660=((g0*g0+4.0*g1*rrs660).^(1.0/2.0)-g0)/(2.0*g1);

    
    
    for i=1:row
        for j=1:col
                      
            Rrs=[u485(i,j),u555(i,j),u660(i,j)];
%             %保存为mat格式
            save('Rrs.mat','Rrs')
           if Rrs(1)==0
           c(i,j)=0;
           ag440(i,j)==0;
       else
            t1=((0.05*(Rrs(1)/Rrs(2))^1.7)/0.06)^(1/0.65);
            t2=0.06*t1^0.65*1.5;
            x0 = [t1,t2];
            lb=[0.0001,0.001];
            ub=[5,5];
            [y,resnorm] = lsqnonlin(@funQAA1,x0,lb,ub);
            c(i,j)=y(1);
            ag440(i,j)=y(2);
            disp(i/row*100);
           end   
        end
        
    end
    

figure;
c(c>0.5)=0;
x=real(c);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

clims=[0,0.5];
C=imagesc(x,clims);
colormap default;
colormap(gca,jet);
axis off;
colorbar


set(C,'alphadata',x~=0);
axis image;
axis off ;


figure;
ag440(ag440>0.35)=0;
ag440(ag440<0.01)=0;
y=real(ag440);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

clims=[0.01,0.35];
C=imagesc(y,clims);
colormap(gca,jet);
axis off;
colorbar


set(C,'alphadata',y~=0);
axis image;
axis off ;
