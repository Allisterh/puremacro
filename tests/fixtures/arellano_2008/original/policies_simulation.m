
% This program uploads vectors saved from fortran and computes simulation
% statistics and generates graphs

clear all;

%% ss: number of shocks ; q: number of bond grid points

ss=21;
n=200;
q;

%% Upload vectors 

sav=dlmread('c:\Arellano\default\sav.f90','');
[i,j]=size(sav);
if j>3
    sav=sav(:,1:3);
end
[i,j]=size(sav);
sav=reshape(sav',i*j,1);
sav=sav(1:q*ss);
[i,j]=size(sav);
sav=reshape(sav, i*j/ss,ss);

savnd=dlmread('c:\Arellano\default\savnd.f90','');
[i,j]=size(savnd);
if j>3
    savnd=savnd(:,1:3);
end
[i,j]=size(savnd);
savnd=reshape(savnd',i*j,1);
savnd=savnd(1:q*ss);
[i,j]=size(savnd);
savnd=reshape(savnd, i*j/ss,ss);

b=dlmread('c:\Arellano\default\b.f90','');
[i,j]=size(b);
if j>3
    b=b(:,1:3);
end
[i,j]=size(b);
b=reshape(b',i*j,1);
b=b(1:n);

v=dlmread('c:\Arellano\default\v.f90','');
[i,j]=size(v);
if j>3
    v=v(:,1:3);
end
[i,j]=size(v);
v=reshape(v',i*j,1);
v=v(1:q*ss);
[i,j]=size(v);
v=reshape(v, i*j/ss,ss);

p=dlmread('c:\Arellano\default\p.f90','');
[i,j]=size(p);
if j>3
    p=p(:,1:3);
end
[i,j]=size(p);
p=reshape(p',i*j,1);
p=p(1:q*ss);

[i,j]=size(p);
p=reshape(p, i*j/ss,ss);

psav=dlmread('c:\Arellano\default\psav.f90','');
[i,j]=size(psav);
if j>3
    psav=psav(:,1:3);
end
[i,j]=size(psav);
psav=reshape(psav',i*j,1);
psav=psav(1:q*ss);
[i,j]=size(psav);
psav=reshape(psav, i*j/ss,ss);

psavnd=dlmread('c:\Arellano\default\psavnd.f90','');
[i,j]=size(psavnd);
if j>3
    psavnd=psavnd(:,1:3);
end
[i,j]=size(psavnd);
psavnd=reshape(psavnd',i*j,1);
psavnd=psavnd(1:q*ss);
[i,j]=size(psavnd);
psavnd=reshape(psavnd, i*j/ss,ss);

i_s_star=dlmread('c:\Arellano\default\i_s_star.f90','');
[i,j]=size(i_s_star);
if j>3
    i_s_star=i_s_star(:,1:3);
end
[i,j]=size(i_s_star);
i_s_star=reshape(i_s_star',i*j,1);
i_s_star=i_s_star(1:q*ss);
[i,j]=size(i_s_star);
i_s_star=reshape(i_s_star, i*j/ss,ss);

def=dlmread('c:\Arellano\default\def.f90','');
[i,j]=size(def);
if j>3
    def=def(:,1:3);
end
[i,j]=size(def);
def=reshape(def',i*j,1);
def=def(1:q*ss);
[i,j]=size(def);
def=reshape(def, i*j/ss,ss);

c=dlmread('c:\Arellano\default\cons.f90','');
[i,j]=size(c);
if j>3
    c=c(:,1:3);
end
[i,j]=size(c);
c=reshape(c',i*j,1);
c=c(1:q*ss);
[i,j]=size(c);
c=reshape(c, i*j/ss,ss);

yy=dlmread('c:\Arellano\default\yy.f90','');
[i,j]=size(yy);
if j>3
    yy=yy(:,1:3);
end
[i,j]=size(yy);
yy=reshape(yy',i*j,1);
yy=yy(1:q*ss);
[i,j]=size(yy);
yy=reshape(yy, i*j/ss,ss);

dprob=dlmread('c:\Arellano\default\dprob.f90','');
[i,j]=size(dprob);
if j>3
    dprob=dprob(:,1:3);
end
[i,j]=size(dprob);
dprob=reshape(dprob',i*j,1);
dprob=dprob(1:q*ss);
[i,j]=size(dprob);
dprob=reshape(dprob, i*j/ss,ss);

conaut=dlmread('c:\Arellano\default\conaut.f90','');
[i,j]=size(conaut);
if j>3
    conaut=conaut(:,1:3);
end
[i,j]=size(conaut);
conaut=reshape(conaut',i*j,1);
conaut=conaut(1:ss);

izero=dlmread('c:\Arellano\default\izero.f90','');
izero=izero(1,1);
theta=dlmread('c:\Arellano\default\theta.f90','');
theta=theta(1,1);

%Bond positions where 0<price<1/r

posdef=zeros(q,ss);
posdef=1.*(p>=1/1.01)+2.*(p<=0.00001);  

[i,j]=find(posdef==0);
lbb=min(i)-1;
ubb=max(i)+1;
if lbb==0
    lbb=1;
end

% FIGURES
% SAVINGS AND VALUE FUNCTION
figure
subplot(1,2,1)
plot(b/10,[savnd(:,8)/10 savnd(:,12)/10],'LineWidth',2)
title({'{\itSavings Function}','{\itB''(B,y)}'},'FontSize',14)
legend('{\ity_{Low}}' , '{\ity_{High}}','Location','Northwest')
xlabel('{\itB}','FontSize',14)
subplot(1,2,2)
plot(b/10,[v(:,8) v(:,12)],'LineWidth',2)
title({'{\itValue Function}','{\itv^o(B,y)}'},'FontSize',14)
legend('{\ity_{Low}}' , '{\ity_{High}}','Location','Northwest')
xlabel('{\itB}','FontSize', 14)


%   BOND PRICE FUNCTIONS
ini=100;
figure
subplot(1,2,1)
plot(b(1:izero)/10,[p(1:izero,8) p(1:izero,12)],'LineWidth',2)
xlabel('{\itB''}','FontSize',14)
legend('{\ity_{Low}}' , '{\ity_{High}}','FontSize', 14 ,'Location','Southwest')
title({'{\itBond Price Schedule}','{\itq(B'',y)}'},'FontSize',14)
subplot(1,2,2);
plot(b(ini:n)/10,(1./psav(ini:n,8)).^4-1, b(ini:n)/10, (1./psav(ini:n,12)).^4-1,'LineWidth',2)
xlabel('{\itB}','FontSize',14)
legend('{\ity_{Low}}' , '{\ity_{High}}','FontSize', 14 ,'Location','Northeast')
title({'{\itEquilibrium Interest Rate}','{\it1/q(B''(B,y),y)}'},'FontSize',14)


%SIMULATIONS

%% Business cycle statistics 


endtime=500000;
T=endtime;

k=1;
if k==1;
    P=dlmread('c:\Arellano\default\PIMAT.dat', '') ;
    Snew       = markovchain(P,T,1); 
else    
    %load Snew Snew;
end

rshoc=1.017*ones(ss,1);

%Simulating the economy by feeding optimal decision rules
dec=zeros(endtime,1);
css=zeros(endtime,1);
bss=zeros(endtime,1);
defss=zeros(endtime,1);
pss=zeros(endtime,1);
pssnd=zeros(endtime,1);
yss=zeros(endtime,1);
countdef=zeros(endtime,1);
tbss=zeros(endtime,1);
swit=zeros(endtime,1);
rrr=zeros(endtime,1);
income=zeros(endtime,1);
asset=zeros(endtime,1);
defdist=zeros(endtime,1);
rfss=zeros(endtime,1);

dec(1)=q;

for i=2:endtime   
    if defss(i-1)==0;
        
        m=dec(i-1);
    for j=1:ss;
        
    if Snew(i)==j
        
        if def(m,j)==0; %stay in nondefault distribution
            dec(i)=i_s_star(m,j);
            asset(i)=bss(i-1);
            bss(i)=sav(m,j);
            yss(i)=yy(m,j);
            css(i)=c(m,j);
            defss(i)=0;
            pss(i)=psav(m,j);
            pssnd(i)=psavnd(m,j);     
            tbss(i)=(yss(i)-css(i))/yss(i);
            income(i)=yss(i)+bss(i-1);
            pvc=css(i-1)+pss(i)*css(i);
            pvy=yss(i-1)+pss(i)*yss(i);           
            rfss(i)=rshoc(j);
            
        else 
       
            dec(i)=izero;
            defss(i)=1;%move to default distribution
            countdef(i)=1;
            asset(i)=bss(i-1);
            bss(i)=0;
            pss(i)=0;
            pssnd(i)=psavnd(m,j);
            rrr(i)=pssnd(i);           
            css(i)=conaut(j);
            yss(i)=conaut(j);
            tbss(i)=(yss(i)-css(i))/yss(i);
            income(i)=yss(i)+bss(i-1);
            defss(i)=1;%move to default distribution
            countdef(i)=1;
            pvc=css(i-1)+pss(i)*css(i);
            pvy=yss(i-1)+pss(i)*yss(i);
            rfss(i)=rshoc(j);
        
        end
          
    end
end


elseif defss(i-1)==1;
    dec(i)=izero;
    defdist(i)=1;
    for j=1:ss;
    if Snew(i)==j
        
         bss(i)=0;
         pss(i)=0;
         asset(i)=0;
         rfss(i)=rshoc(j);
         css(i)=conaut(j);
         yss(i)=conaut(j);
         tbss(i)=(yss(i)-css(i))/yss(i);
         income(i)=yss(i)+bss(i-1);
            if rand<theta
            defss(i)=0;
            else
            defss(i)=1;
            end
    end
    end    
end
end
 
%% Cut initial draws 
bas=1000;
bss1=bss(bas:endtime);
css1=css(bas:endtime);
yss1=yss(bas:endtime);
pss1=pss(bas:endtime);
defss1=defss(bas:endtime);
tbss1=tbss(bas:endtime);
income1=income(bas:endtime);
asset1=asset(bas:endtime);
defdist1=defdist(bas:endtime);
rfss1=rfss(bas:endtime);

%Calculate simulated data from default states 
[rowsdel]=find(pss1==0);
ndd=size(rowsdel);
cssdef=css1(rowsdel);
cssdef=cssdef(2:ndd);
defssdef=defss1(rowsdel);
defssdef=defssdef(2:ndd);
yssdef=yss1(rowsdel);
yssdef=yssdef(2:ndd);

 
% Policy functions from Limiting Distribution Conditional on not Defaulting
css1(rowsdel)=[];
bss1(rowsdel)=[];
pss1(rowsdel)=[];
yss1(rowsdel)=[];
tbss1(rowsdel)=[];
income1(rowsdel)=[];
asset1(rowsdel)=[];
rfss1(rowsdel)=[];

% Default probability 
'default prob'
1-(1-sum(countdef)/(endtime-ndd(1)))^4

%% Finding DEFAULT EPISODES in limiting distribution 
'number of defaults'
defepi=sum(countdef) 
[defepirows]=find(countdef==1);

% Stats during the period when default is chosen 

cdf=zeros(defepi,1);
ydf=zeros(defepi,1);
bdf=zeros(defepi,1);
tbdf=zeros(defepi,1);
pdf=zeros(defepi,1);
rfdf=zeros(defepi,1);
spdf=zeros(defepi,1);
yyy=mean(log(yss1));
ccc=mean(log(css1));
bbb=mean(bss1);
tbtb=mean(tbss1);

for i=1:defepi
cdf(i)=log(css(defepirows(i)))-ccc;
ydf(i)=log(yss(defepirows(i)))-yyy;
bdf(i)=(bss(defepirows(i)-1));
tbdf(i)=-tbss(defepirows(i)-1);
spdf(i)=(1./rrr(defepirows(i)))^4-rfss(defepirows(i))^4;
pdf(i)=(1./rrr(defepirows(i)))^4;
end

% Statitics for period default
 'Default period : y c b tb spread p '
 'mean'
 xxdf= [ydf  cdf bdf/mean(yss1) tbdf  spdf pdf ];
 mean(xxdf)
 std(xxdf)

% Generate time series containing a default.  
n=74; % number of obervations to include prior to default 

cstd=zeros(defepi,1);
ystd=zeros(defepi,1);
bstd=zeros(defepi,1);
tbstd=zeros(defepi,1);
pstd=zeros(defepi,1);
spstd=zeros(defepi,1);
bmean_def=zeros(defepi,1);
ymean_def=zeros(defepi,1);

pmean=zeros(defepi,1);
spmean=zeros(defepi,1);
ycorr=zeros(defepi,6);
pcorr=zeros(defepi,6);
autc=zeros(defepi,6);


time=[ones(n,1)];

for k=2:300
   
    i=defepirows(k);
    if k==1
        defpt=1;
    else
        defpt=defepirows(k-1);
    end
    
    if min(pss(i-n:i-1))>0
    %   p bond
    xx5=((1./pss(i-n:i-1)).^4);
    pmean(k)=mean(xx5);
    min(xx5);
    a1=cov(xx5(2:n),xx5(1:n-1))/var(xx5);
    autc(k,5)=a1(2,1);   
    pstd(k)=std(xx5);    
    %   spread bond
    xx6=((1./pss(i-n:i-1)).^4-(rfss(i-n:i-1)).^4);
    spmean(k)=mean(xx6);
    min(xx6);
    a1=cov(xx6(2:n),xx6(1:n-1))/var(xx6);
    autc(k,6)=a1(2,1);   
    spstd(k)=std(xx6);
    % y
    xx1=yss(i-n:i-1);   
    ymean_def(k)=mean(xx1);
    xx1=log(xx1);
    beta=inv(time'*time)*time'*xx1; 
    xx1=xx1-time*beta;
    a1=cov(xx1(2:n),xx1(1:n-1))/var(xx1);
    autc(k,1)=a1(2,1);
    ystd(k)=std(xx1);
    % css
    xx2=css(i-n:i-1); 
    xx2=log(xx2);
    beta=inv(time'*time)*time'*xx2; 
    xx2=xx2-time*beta;
    a1=cov(xx2(2:n),xx2(1:n-1))/var(xx2);
    autc(k,2)=a1(2,1);
    cstd(k)=std(xx2);
    % b   
    xx3=bss(i-n:i-1);
    bmean_def(k,1)=mean(xx3);
    beta=inv(time'*time)*time'*xx3; 
    xx3=xx3-time*beta;
    a1=cov(xx3(2:n),xx3(1:n-1))/var(xx3);
    autc(k,3)=a1(2,1);    
    bstd(k)=std(xx3);
    % tb
    xx4=tbss(i-n:i-1); 
    %tbstd(k)=std(xx4);
    %beta=inv(time'*time)*time'*xx4; 
    %xx4=xx4-time*beta;
    a1=cov(xx4(2:n),xx4(1:n-1))/var(xx4(1:n));
    kkk=1;
    if var(xx4)==0
        autc(k,4)=0;
        kkk=0;
        tbstd(k)=0;
    else
    autc(k,4)=a1(2,1);
        kkk=1;
        tbstd(k)=std(xx4);
    end

    xxx=[xx1 xx2 xx3 xx4 xx5 xx6] ;   
    zz=corrcoef(xxx);
    ycorr(k,:)=zz(:,1)';
    if std(xx5)>0; 
    pcorr(k,:)=zz(:,6)';
    end
else
    k=k+1;
end

end

%% Deleting episodes with multiple defaults 

 [rowsdf]=find(ystd==0);
 cstd(rowsdf)=[];
 ystd(rowsdf)=[];
 bstd(rowsdf)=[];
 tbstd(rowsdf)=[];
 pstd(rowsdf)=[];
 spstd(rowsdf)=[];
 bmean_def(rowsdf)=[];
 ymean_def(rowsdf)=[];
 pmean(rowsdf)=[]; 
 spmean(rowsdf)=[]; 
 ycorr(rowsdf,:)=[];
 pcorr(rowsdf,:)=[];

 %%  Calculate averages of standard deviations and correlations across
 %%  default episodes 
 
'y c b tb p spread'
'mean of std '
[mean(ystd) mean(cstd) mean(bstd) mean(tbstd) mean(pstd) mean(spstd)]
'std'
[std(ystd) std(cstd) std(bstd) std(tbstd) std(pstd) std(spstd)]
'mean b'
mean(bmean_def./ymean_def)
'Y-corr'
'mean'
mean(ycorr) 
'std'
std(ycorr)
'r-corr'
'mean'
mean(pcorr)
'std'
std(pcorr)
'mean spread'
'mean'
mean(spmean)
'std'
std(spmean)
'default rate'
1-(1-sum(countdef)/(endtime-ndd(1)))^4
'# defaults included'
size(ystd)


%%  Simulation Default Event 

ss=21;
ymat=dlmread('c:\Arellano\default\YMAT.dat','');
ymat=ymat';
[i,m]=size(ymat);
ydt=dlmread('c:\Arellano\default\ydt.txt','');
[k,m]=size(ydt);
datgrid=zeros(k,1);
% find the row in ymat tha is closest to ydt 
for j=1:k
    v0=ymat(1)-ydt(j);
    datgrid(j)=1;
    for jj=1:i
        v1=ymat(jj)-ydt(j);
        if sum(abs(v1))<sum(abs(v0))
            datgrid(j)=jj;
            v0=v1;
       end
    end
end
Snew=datgrid;
Snew=[1;Snew];
[endtime,oo]=size(Snew);
T=endtime;
%Feeding in actual income shocks and feeding optimal decision rules
dec=zeros(endtime,1);
css=zeros(endtime,1);
bss=zeros(endtime,1);
defss=zeros(endtime,1);
pss=zeros(endtime,1);
pssnd=zeros(endtime,1);
yss=zeros(endtime,1);
countdef=zeros(endtime,1);
tbss=zeros(endtime,1);
swit=zeros(endtime,1);
dec(1)=izero;
for i=2:endtime   
    if defss(i-1)==0;        
        m=dec(i-1);
    for j=1:ss;        
    if Snew(i)==j        
        if def(m,j)==0; %stay in nondefault distribution
            dec(i)=i_s_star(m,j);
            bss(i)=sav(m,j);
            yss(i)=yy(m,j);
            css(i)=c(m,j);
            defss(i)=0;
            pss(i)=psav(m,j);
            pssnd(i)=psavnd(m,j);     
            tbss(i)=(yss(i)-css(i))/yss(i);
        else 
            dec(i)=izero;
            defss(i)=1;%move to default distribution
            countdef(i)=1;
            bss(i)=0;
            pss(i)=0;
            pssnd(i)=psavnd(m,j);
            rrr=pssnd(i);           
            css(i)=conaut(j);
            yss(i)=conaut(j);
            tbss(i)=(yss(i)-css(i))/yss(i);
            defss(i)=1;%move to default distribution  
            countdef(i)=1;
        end  
    end
    end
elseif defss(i-1)==1;
    dec(i)=izero;
    for j=1:ss;
    if Snew(i)==j 
         bss(i)=0;
         pss(i)=0;
         pssnd(i)=rrr;
         css(i)=conaut(j);
         yss(i)=conaut(j);
         tbss(i)=(yss(i)-css(i))/yss(i);
            if rand<1
            defss(i)=0;
            else
            defss(i)=1;
            end
    end     
    end    
end
end
bas=1;
 
bss1=bss(bas:endtime);
css1=css(bas:endtime);
yss1=yss(bas:endtime);
pss1=pss(bas:endtime);
defss1=defss(bas:endtime);
tbss1=tbss(bas:endtime);

[rowsdel]=find(pss1==0);
css1(rowsdel)=[];
bss1(rowsdel)=[];
pss1(rowsdel)=[];
yss1(rowsdel)=[];
tbss1(rowsdel)=[];
time=[2:1:endtime]';
ylog=[yss-mean(yss1)];
clog=css-mean(css1);

figure
plotyy(time, [(1./pss(2:endtime)).^4-1.017^4],time, [ylog(2:endtime) clog(2:endtime)])
title ('Argentina Time Series')





