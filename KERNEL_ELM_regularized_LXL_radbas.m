function [max_test_gmean,max_test_ova,X,Y,AUC_soft,AUC_hard,F_measure,TrainingTime, TestingTime] = UGECSKELM(TrainingData_File, TestingData_File, Elm_Type, Kernel_type,C,C1,Kernel_para)

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
train_data = load(TrainingData_File);
T=train_data(:,1)';
T1=T;
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array


%%%%%%%%%%% Load testing dataset
test_data =  load(TestingData_File);
TV.T=test_data(:,1)';
TV.T1=TV.T;
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                        %   Release raw testing data array


NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);




if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;

end                                                 %   end if of Elm_Type

for i=1: length(label)
       E(i)=sum(T1==label(i));
       LBLIDX(i,:)=(T1==label(i));
end
AVGIPC=mean(E);
[MINIPC  MINL]=min(E);
TDIDX=1:NumberofTrainingData;

tic;

P_POS=P(:,LBLIDX(MINL,:));
T_POS=T(:,LBLIDX(MINL,:));
P_NEG=P(:,~LBLIDX(MINL,:));
T_PNEG=T(:,~LBLIDX(MINL,:));
%FIND NUMBER OF UNIVERSUM POINTS and GENERATE
UNIV_NO=ceil(0.5*NumberofTrainingData);

for k=1:UNIV_NO
    pidx=randi(size(P_POS,2));
    nidx=randi(size(P_NEG,2));
    TempA=P(:,pidx);
    TempB=P(:,nidx);
    UNIV(k,:)=(TempA+TempB)/2;
end

%MODIFY THE TRAINING DATA TO INCLUDE UNIVERSUM POINTS
P=[P UNIV'];
T=[T zeros(2,UNIV_NO)];
T1=[T1 zeros(1,UNIV_NO)];

for i=1: length(label)
       E(i)=sum(T1==label(i));
       labelidx(i,:)=(T1==label(i));
end
  
  n = size(P,2);
P=P';
NTD=size(P,1);
T=T';
j=1;
for j=1:size(labelidx,1)
for kl=1:size(labelidx,1)
    KM{kl,j} = kernel_matrix(P(labelidx(kl,:),:),Kernel_type, Kernel_para,P(labelidx(j,:),:));
end
   CC(j)=((NTD-E(j))/NTD)*C;
 %CC(j)=1/E(j)*C;
   TC{j}=T(labelidx(j,:),:);
end
SI=0;

d=zeros(number_class,number_class);
for j=1:size(labelidx,1)
    SI=SI+1/CC(j);
for kl=1:size(labelidx,1)
    for op=1:number_class
       d(j,kl)=d(j,kl)+CC(j)/CC(op);
    end
end
end
for j=1:size(labelidx,1)
    TC{j}=d(j,j)* TC{j};
for kl=1:size(labelidx,1)
     KM{kl,j}= d(kl,j)* KM{kl,j};
end
end

 P1=[];
 for kl=1:size(labelidx,1)
P1=[P1
    P(labelidx(kl,:),:)];
end

Omega_train = kernel_matrix(P,Kernel_type, Kernel_para);

 balanced=1;
% regularization ELM
m_vec = mean(Omega_train,2);
if balanced==1,  % if you assume that your dataset is balanced 
	diff_mat = Omega_train - m_vec*ones(1,size(Omega_train,2));
	S = (diff_mat*diff_mat') / size(Omega_train,2); 
else
	for cl=1:number_class
		currSamples = Omega_train(:,find(T1==cl));
		currDiff = currSamples - m_vec*ones(1,size(currSamples,2));
		S = S + (currDiff*currDiff') / size(currSamples,2);
	end
end

KMF=cell2mat(KM);
SI=((C1*SI)*(S*Omega_train)+speye(n)*SI);
II = eye(size(P,1),size(KMF,1));
III=eye(size(KMF,1),size(P,1));
KMF1=II*KMF*III;
IM=SI+speye(n)*KMF1;

% TM1=cell2mat(TC');
% IV = eye(size(T,1),size(TM1,1));

OutputWeight=IM\T;

Y=(Omega_train * OutputWeight)';                             %   Y: the actual output of the training data

TrainingTime=toc;



%%%%%%%%%%% Calculate the output of testing input
tic;
Omega_test = kernel_matrix(P,Kernel_type, Kernel_para,TV.P');
TY=(Omega_test' * OutputWeight)';                            %   TY: the actual output of the testing data
TestingTime=toc;



    
    [a b]=max(TV.T);
    [c d]=max(TY);
    
%     if isnan(c)
%         keyboard;
%     end
    
    tesconf=confusionmat(b,d);
    
   
  max_test_ova= sum(diag(tesconf))/sum(sum(tesconf));
%     tes_avg(TRIAL) = sum(diag(tesconf)./sum(tesconf,2))/NOP;
  max_test_gmean =power(prod(diag(tesconf)./sum(tesconf,2)),1/number_class);


R=sum(tesconf(MINL,MINL))/sum(tesconf(MINL,:));
Per=sum(tesconf(MINL,MINL))/sum(tesconf(:,MINL));
F_measure=2*Per*R/(Per+R);

SSM=TY(MINL,:);
%SSM=squeeze(SM);

SSM(SSM<0)=0;
if ~(sum(SSM)==0)
    SSM=SSM/max(SSM);
end
final_score=SSM;
final_score(isnan(final_score))=0;
resp=TV.T1==label(MINL);
if sum(final_score)==0
    X=0;
    Y=0;
    AUC_soft=0;
else
[X,Y,Tlog,AUC_soft] = perfcurve(resp,final_score,'true');
end 
% AUC_hard=(1+R-((sum(tesconf(~(label==label(MINL)),MINL))))/sum(tesconf(~(label==label(MINL)),:)))/2;
AUC_hard=1;

% plot decision boundary
% X=meshgrid([-1:0.01:1],[-1:0.01:1])
% Y=X';
% D=zeros(40401,2);
% k=1;
% for i=1:201
% for j=1:201
% D(k,:)= [X(i,j) Y(i,j)];
% k=k+1;
% 
% end
% end
% % D=D';
% Omega_D = kernel_matrix(P',Kernel_type, Kernel_para,D);
% DY=(Omega_D' * OutputWeight);
% [c1 d1]=max(DY);
% idx1=d1==1;
% idx2=d1==2;
% D1=D(idx1,:)
% plot(D1(:,1),D1(:,2),'o');
% D2=D(idx2,:)
% hold on
% plot(D2(:,1),D2(:,2),'+');



end

   
  
 
%%%%%%%%%%%%%%%%%% Kernel Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);

% nb_data=10;
if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./kernel_pars(1));
    end
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end


end
