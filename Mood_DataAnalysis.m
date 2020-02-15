

for n=1:n_subjects


	% T = array2table(placdata(n).experiment_data(:,1:10));
	T = array2table(dopadata(n).experiment_data(:,1:10));

	T.Properties.VariableNames(1:10) = {'TrialNum','SideOfScreen','Safe','BigRisky','SmallRisky','SideChosen','Choice','Outcome','RT','Happiness'};
	% cd("C:\Users\rniyogi\Dropbox\Postdoc_UCL\DATA\rlab_incomplete_rewardSWB_code\by_RN\")
	% dirpath = (strcat(['placdata\subject_num_',num2str(placdata(n).subjectnumber)]));


	%% ###### OS ##-------
	cd('/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code/by_RN')
	dirpath = (strcat(['dopadata/subject_num_',num2str(dopadata(n).subjectnumber)]));


	mkdir(dirpath);
	cd(dirpath)
	% writetable(T,'experiment_data.csv');
	writetable(T,'dopa_experiment_data.csv');


end

for n=1:n_subjects

	% result = modelfit_pt(placdata(n));
	% result = modelfit_pt_RKN(placdata(n),150);

	%%% combining datasets %%%%


	stop = 150;
	% result = modelfit_pt_RKN([placdata(n).behavedata(3:stop+2,:) ; dopadata(n).behavedata(3:stop+2,:)],150);

	% result = modelfit_pt_RKN([placdata(n).behavedata(stop+2:end,:) ; dopadata(n).behavedata(stop+2:end,:)],150);

 	% result = modelfit_pt_RKN([placdata(n).behavedata([3:stop/3+2, 2*stop/3+2:stop,1+ stop + stop/3: 1+stop + 2*stop/3],:) ; dopadata(n).behavedata([3:stop/3+2, 2*stop/3+2:stop,1+ stop + stop/3: 1+stop + 2*stop/3],:)],150);

 	%%%% Chunking 

 % 	end_chunk = 6; start_chunk = 1; chunk_size= end_chunk-start_chunk + 1;
	% out = repmat([start_chunk:end_chunk],30,1) +  repmat([0:10:290]',1,chunk_size); out = reshape(out',numel(out),1);

	end_chunk = 10; start_chunk = 1; chunk_size= end_chunk-start_chunk + 1;
	out = repmat([start_chunk:end_chunk],30/2,1) +  repmat([0:2*10:290]',1,chunk_size); out = reshape(out',numel(out),1);


 	result = modelfit_pt_RKN([placdata(n).behavedata(out+2,:); dopadata(n).behavedata(out+2,:)]);

	% cd("C:\Users\ritwik7\Dropbox\Postdoc_UCL\DATA\rlab_incomplete_rewardSWB_code\by_RN\")

	% cd('/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code')


	cd('C:\Users\rniyogi\Dropbox\Postdoc_UCL\DATA\rlab_incomplete_rewardSWB_code\by_RN')

	% dirpath = (strcat(['placdata\subject_num_',num2str(placdata(n).subjectnumber)]));

	%% ####### Blocking
	% dirpath = (strcat(['ActualDataFitting\Pretraining\vcheck\subject_num_',num2str(placdata(n).subjectnumber)]));

	%%% ####### Chunking 
	dirpath = (strcat(['ActualDataFitting\Pretraining\v10chunks\subject_num_',num2str(placdata(n).subjectnumber)]));

	cd(dirpath)
	% save PT_result result
	% save PT_result_train150 result
	% save PT_result_train150_last result

	% save PT_result_combined1sthalf result
	% save PT_result_combined2ndhalf result

	save PT_result_50_splits_combined_1sthalf result
end


% for n=11:26
for n=28:41

	cd('C:\Users\rniyogi\Dropbox\Postdoc_UCL\DATA\rlab_incomplete_rewardSWB_code\by_RN')

	dirpath = (strcat(['placdata/subject_num_',num2str(n),'/']));

	%%%%%% Blocking
	% dirpath = (strcat(['ActualDataFitting/Pretraining/vcheck/subject_num_',num2str(n),'/']));
	
	%%%% Chunking
	dirpath = (strcat(['ActualDataFitting/Pretraining/v10chunks/subject_num_',num2str(n),'/']));


	cd(dirpath)
	% load PT_result.mat;
	% load PT_result_train150.mat;
	% load PT_result_train150_last.mat

	% load PT_result_combined1sthalf result
	% load PT_result_combined2ndhalf result

	load PT_result_50_splits_combined_1sthalf result



	PT_loss(n-10) = - result.modelLL;
	PT_pseudoR2(n-10) = result.pseudoR2;


 	PT_accuracy(n-10) = (sum((1-result.data(:,4)).*(1-result.probchoice)) + sum(result.data(:,4).*(result.probchoice)) ) / length(result.probchoice);

 	%%% same as 
	% PT_accuracy(n-10) = (sum(result.probchoice(result.data(:,4)==1)) + sum(1-result.probchoice(result.data(:,4)==0)))/length(result.probchoice);
	%%%%%%%%%%%%%%%

	% [loglike, utildiff, logodds, probchoice] = model_param_RKN(result.b,placdata(find([placdata.subjectnumber] ==n)),150);

	% test_data.behavedata = [placdata(find([placdata.subjectnumber] ==n)).behavedata(3:stop/2+2,:); dopadata(find([dopadata.subjectnumber] ==n)).behavedata(3:stop/2+2,:)];

	% test_data.behavedata = [placdata(find([placdata.subjectnumber] ==n)).behavedata(stop+3:stop+3+stop/2-1,:); dopadata(find([dopadata.subjectnumber] ==n)).behavedata(stop+3:stop+3+stop/2-1,:)];

	%%%% ---blocking 
	% test_data.behavedata = [placdata(find([placdata.subjectnumber] ==n)).behavedata([stop/3+2:stop/3+1+stop/6, stop+1:stop + stop/6 ,2+stop + 2*stop/3: 1+stop + 2*stop/3 + stop/6],:); dopadata(find([dopadata.subjectnumber] ==n)).behavedata([stop/3+2:stop/3+1+stop/6, stop+1:stop + stop/6 ,2+stop + 2*stop/3: 1+stop + 2*stop/3 + stop/6],:)];


	%%%% --- chunking ----
	% end_chunk = 8; start_chunk = 7; chunk_size= end_chunk-start_chunk + 1;
	% out = repmat([start_chunk:end_chunk],30,1) +  repmat([0:10:290]',1,chunk_size); out = reshape(out',numel(out),1);

	
	
	end_chunk = 15; start_chunk = 11; chunk_size= end_chunk-start_chunk + 1;
	out = repmat([start_chunk:end_chunk],30/2,1) +  repmat([0:2*10:290]',1,chunk_size); out = reshape(out',numel(out),1);


	test_data.behavedata = [placdata(find([placdata.subjectnumber] ==n)).behavedata(out+2,:); dopadata(find([dopadata.subjectnumber] ==n)).behavedata(out+2,:)];

	size(test_data)

	[loglike, utildiff, logodds, probchoice_test] = model_param_RKN(result.b,test_data);
	
	

	% PT_pseudoR2_test(n-10) = 1 + loglike/(log(0.5)*stop);

	PT_pseudoR2_test(n-10) = 1 + (-(sum((1-test_data.behavedata(:,7)).*log(1-probchoice_test)) + sum(test_data.behavedata(:,7).*log(probchoice_test)) ) / length(probchoice_test))/log(0.5)


	% PT_pseudoR2_test(n-10) = 1 + loglike/(log(0.5)*75);

	% [loglike, utildiff, logodds, probchoice] = model_param(result.b,dopadata(find([dopadata.subjectnumber] ==n)));
	% PT_pseudoR2_test(n-10) = 1 + loglike/(log(0.5)*300);

	PT_loss_test(n-10) = loglike;




	PT_accuracy_test(n-10) = (sum((1-test_data.behavedata(:,7)).*(1-probchoice_test)) + sum(test_data.behavedata(:,7).*(probchoice_test)) ) / length(probchoice_test) ;
	% PT_accuracy_test(n-10) =(sum(probchoice_test(test_data.behavedata(:,7)==1)) + sum(1-probchoice_test(test_data.behavedata(:,7)==0)))/length(probchoice_test);


	T = array2table(result.probchoice);
	T.Properties.VariableNames(1) = {'probchoice_train'};

	S = array2table(probchoice_test);
	S.Properties.VariableNames(1) = {'probchoice_test'};

	% writetable(T,'PT_probchoice.csv')
	% writetable(T,'PT_probchoice_TEST_Sess.csv')
	% writetable(T,'PT_probchoice_TEST_Sess_trainLast.csv')

	% writetable(T,'PT_probchoice_combined_1sthalf.	csv')

	% writetable(T,'PT_probchoice_combined_2ndhalf.csv')

	writetable(T,'PT_probchoice_50_split_combined_1sthalf.csv')
	writetable(S,'PT_probchoice_test_50_split_combined_1sthalf.csv')


	cd ../..

end

T = array2table([[11:41]', PT_loss', PT_pseudoR2',PT_accuracy',PT_loss_test',PT_pseudoR2_test',PT_accuracy_test']);
T.Properties.VariableNames(1:7) = {'Subject_number','PT_loss','PT_pseudoR2','PT_accuracy','PT_loss_test','PT_pseudoR2_test','PT_accuracy_test'};
% writetable(T,'PT_loss_updated.csv')
% writetable(T,'PT_loss_updated_TESTSinSESS.csv')
% writetable(T,'PT_loss_updated_combined_1sthalf.csv')
% writetable(T,'PT_loss_updated_combined_2ndhalf.csv')

writetable(T,'PT_loss_updated_50_split_combined_1sthalf.csv')


% ################################

load('PT_result.mat');
T = array2table(result.probchoice);
T.Properties.VariableNames(1) = {'probchoice_train'};
writetable(T,'PT_probchoice.csv')
% #############################







%% For app data from GBE
load GBEsuperplayers.mat;
n_subjects = size(ritwikGBE,1)

for n=1:n_subjects
	for l = 1:length(ritwikGBE(n).data)



			T = array2table((ritwikGBE(n).data{l}(:,1:10)));
			T.Properties.VariableNames(1:10) = {'TrialNum','SideOfScreen','Safe','BigRisky','SmallRisky','SideChosen','Choice','Outcome','RT','Happiness'};
			cd("C:\Users\rniyogi\Dropbox\Postdoc_UCL\DATA\rlab_incomplete_rewardSWB_code\by_RN\")
			% dirpath = (strcat(['placdata\subject_num_',num2str(placdata(n).subjectnumber)]));


			%% ###### OS ##-------
			% cd('/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code/by_RN')
			dirpath = (strcat(['appdata/subject_num_',num2str(ritwikGBE(n).uid)]));


			mkdir(dirpath);
			cd(dirpath)
			% writetable(T,'experiment_data.csv');
			writetable(T,strcat(['app_data_num_play_',num2str(l),'.csv']));

		end


end


PT_loss =  zeros(1,n_subjects); PT_pseudoR2= zeros(1,n_subjects); PT_accuracy = zeros(1,n_subjects);
PT_loss_test =  zeros(1,n_subjects); PT_pseudoR2_test= zeros(1,n_subjects); PT_accuracy_test = zeros(1,n_subjects);

for n=1:n_subjects

	cd("C:\Users\rniyogi\Dropbox\Postdoc_UCL\DATA\rlab_incomplete_rewardSWB_code\by_RN\")
		
	dirpath = (strcat(['appdata/subject_num_',num2str(ritwikGBE(n).uid)]));
%%%% --- COMNMENT OUT THIS LINE IF NECESSARY -----------
	% dirpath	= strcat([dirpath,'/OddEvenPlays']);
	% dirpath	= strcat([dirpath,'/OddEvenPlays/RandomizedPlays1']);
	dirpath	= strcat([dirpath,'/Play_by_play']);

	% ###########################################

	if exist(dirpath)	
		dirpath
		cd(dirpath)

		train_data = readtable("train_data.csv");
		train_data(:,1)=[];
		train_data = train_data(:,1:16);

		if exist("val_data.csv")~=2
			val_data = readtable("val_data.csv");
			val_data(:,1)=[];

			result = modelfit_pt_RKN([table2array(train_data); table2array(val_data)]);

		else 

			result = modelfit_pt_RKN([table2array(train_data)]);
		end

		save PT_result_50_splits_combined_1sthalf result


		load PT_result_50_splits_combined_1sthalf result



		PT_loss(n) = - result.modelLL;
		PT_pseudoR2(n) = result.pseudoR2;
		PT_accuracy(n) = (sum((1-result.data(:,4)).*(1-result.probchoice)) + sum(result.data(:,4).*(result.probchoice)) ) / length(result.probchoice);

		test_dat = readtable("test_data.csv");
		test_dat(:,1)=[];	
		test_dat = test_dat(:,1:16);
	 	
	 	test_data.behavedata=table2array(test_dat);
		[loglike, utildiff, logodds, probchoice_test] = model_param_RKN(result.b,test_data);
		
		


		PT_pseudoR2_test(n) = 1 + (-(sum((1-test_data.behavedata(:,7)).*log(1-probchoice_test)) + sum(test_data.behavedata(:,7).*log(probchoice_test)) ) / length(probchoice_test))/log(0.5)


		PT_loss_test(n) = loglike;




		PT_accuracy_test(n) = (sum((1-test_data.behavedata(:,7)).*(1-probchoice_test)) + sum(test_data.behavedata(:,7).*(probchoice_test)) ) / length(probchoice_test) ;
		% PT_accuracy_test(n-10) =(sum(probchoice_test(test_data.behavedata(:,7)==1)) + sum(1-probchoice_test(test_data.behavedata(:,7)==0)))/length(probchoice_test);


		T = array2table(result.probchoice);
		T.Properties.VariableNames(1) = {'probchoice_train'};

		S = array2table(probchoice_test);
		S.Properties.VariableNames(1) = {'probchoice_test'};

		writetable(T,'PT_probchoice_50_split_combined_1sthalf.csv')
		writetable(S,'PT_probchoice_test_50_split_combined_1sthalf.csv')

	end
end



T = array2table([[1:n_subjects]', PT_loss', PT_pseudoR2',PT_accuracy',PT_loss_test',PT_pseudoR2_test',PT_accuracy_test']);
T.Properties.VariableNames(1:7) = {'Subject_number','PT_loss','PT_pseudoR2','PT_accuracy','PT_loss_test','PT_pseudoR2_test','PT_accuracy_test'};
writetable(T,'PT_loss_updated_50_split_combined_1sthalf.csv')





























%% For generatting samples from PT
data = placdata(find([placdata.subjectnumber] ==29)).behavedata;
certain = data(:,3);
gainVal = data(:,4);
lossVal = data(:,5);
choice = data(:,7);


%%%%%%%%%%%%%%
clear idx_
num_combs=10
for t=1:num_combs

idx_(t,:) = randperm(300);
end

idx= reshape(idx_,num_combs*300,1);

idx(idx==1)=[];
%%%%%%%%%%%%%%%%%%%%%

certain = data(idx,3);
gainVal = data(idx,4);
lossVal = data(idx,5);


% alphaPlus = result.b(3);
% alphaMinus = result.b(4);
% lambda = result.b(2);
% mu = result.b(1);



%% Normalized
% certain = data(idx,3)/100;
% gainVal = data(idx,4)/100;
% lossVal = data(idx,5)/100;



%% Usual params
alphaPlus = 0.8;
alphaMinus = 0.8;
lambda = 2;

alphaPlus = 0.9;
alphaMinus = 0.9;
lambda = 1.5;





mu=0.5;
% mu=1;
% mu = 5;
% mu=20;






utilcertain = (certain>0).*abs(certain).^alphaPlus - ...
    (certain<0).*lambda.*abs(certain).^alphaMinus;
winutil       = gainVal.^alphaPlus;
lossutil      = -lambda*(-lossVal).^alphaMinus;
utilgamble    = 0.5*winutil+0.5*lossutil;
utildiff      = utilgamble - utilcertain;
logodds       = mu*utildiff;
probchoice    = 1 ./ (1+exp(-logodds));     %prob of choosing gamble


choice = binornd(1,probchoice);
% data = placdata(find([placdata.subjectnumber] ==29)).behavedata;
% data(1,:) = [];%choice(1)=[];


clear outcome
outcome(choice==0)= data(idx(choice==0),3);

aa = [idx(choice==1) binornd(1,repmat(0.5,length(find(choice==1)),1))+2];
% aa(aa(:,1)==max(aa(:,1)),:) = [];
outcome(choice==1) = data(sub2ind(size(data),aa(:,1),aa(:,2)));


T = array2table([data(idx,[3:5]),choice,outcome',[1:length(choice)]']);
T.Properties.VariableNames(1:6) = {'Safe','BigRisky','SmallRisky','Choice','Outcome','TrialNum'};



%% fitting data with PT


% data=generateddata1500subj29params; stop=750



% #################### 3000
% data=generateddata3000subj29params; stop=1500
% data=generateddata3000mu1params; stop=1500
% data=generateddata3000mu5params; stop=1500
% data=generateddata3000mu20params; stop=1500

% data=generateddata300mu05params; stop=150
data=generateddata600mu05params; stop=300
% data=generateddata1500mu05params; stop=750


% data=generateddata300mu1params; stop=150
% data=generateddata300mu5params; stop=150
% data=generateddata300mu20params; stop=150

% data=generateddata1500mu1params; stop=750
% data=generateddata1500mu5params; stop=750
% data=generateddata1500mu20params; stop=750


% data.Safe = data.Safe/100;
% data.BigRisky = data.BigRisky/100;
% data.SmallRisky = data.SmallRisky/100;
% data.Outcome = data.Outcome/100;



result = modelfit_pt_RKN(data,stop)

save PT_result_600 result
load PT_result_600 result 

% save PT_result_1500 result
% load PT_result_1500 result 

% save PT_result_3000 result
% load PT_result_3000 result 

% save PT_result_300 result
% load PT_result_300 result 




PT_loss = -result.modelLL;
PT_pseudoR2 = result.pseudoR2;

test_data.behavedata = data;



	[loglike, utildiff, logodds, probchoice] = model_param_RKN(result.b,test_data,stop);
	PT_pseudoR2_test = 1 + loglike/(log(0.5)*stop/2);


	PT_loss_test = loglike;

	T = array2table(result.probchoice);
	T.Properties.VariableNames(1) = {'probchoice_train'};
	
	% writetable(T,'PT_probchoice_1500.csv')
	% writetable(T,'PT_probchoice_300.csv')
	writetable(T,'PT_probchoice_600.csv')

	T = array2table([27, PT_loss', PT_pseudoR2',PT_loss_test',PT_pseudoR2_test']);
	T.Properties.VariableNames(1:5) = {'Subject_number','PT_loss','PT_pseudoR2','PT_loss_test','PT_pseudoR2_test'};
	% writetable(T,'PT_loss_updated_1500.csv')
	% writetable(T,'PT_loss_updated_300.csv')
	writetable(T,'PT_loss_updated_600.csv')

