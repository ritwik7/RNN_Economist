

for n=1:n_subjects


	T = array2table(placdata(n).experiment_data(:,1:10));
	T.Properties.VariableNames(1:10) = {'TrialNum','SideOfScreen','Safe','BigRisky','SmallRisky','SideChosen','Choice','Outcome','RT','Happiness'};
	cd("C:\Users\rniyogi\Dropbox\Postdoc_UCL\DATA\rlab_incomplete_rewardSWB_code\by_RN\")
	dirpath = (strcat(['placdata\subject_num_',num2str(placdata(n).subjectnumber)]));
	mkdir(dirpath);
	cd(dirpath)
	writetable(T,'experiment_data.csv');

end

for n=1:n_subjects

	result = modelfit_pt(placdata(n));
	% cd("C:\Users\ritwik7\Dropbox\Postdoc_UCL\DATA\rlab_incomplete_rewardSWB_code\by_RN\")

	cd('/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code')
	dirpath = (strcat(['placdata\subject_num_',num2str(placdata(n).subjectnumber)]));
	cd(dirpath)
	save PT_result result

end


for n=11:41
	dirpath = (strcat(['placdata/subject_num_',num2str(n),'/']));
	cd(dirpath)
	load PT_result.mat;
	PT_loss(n-10) = - result.modelLL;
	cd ../..
end

T = array2table([[11:41]', PT_loss']);
T.Properties.VariableNames(1:2) = {'Subject_number','PT_loss'};
writetable(T,'PT_loss.csv')