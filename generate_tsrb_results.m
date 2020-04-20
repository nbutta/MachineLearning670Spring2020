%%
% File: generate_tsrb_results.m
%   Generate output file used by the GTSRB Analysis Tool
%
% Info:
%   Class: EN.525.670.81 - Machine Learning for Signal Processing
%   Term: Spring 2020
%   Author: Johanna Rivera
%
function generate_tsrb_results(fname,trainInfo,testInfo,pLabels)

    outFid = fopen(fname,'w');
    gtFid = fopen('GT-test_data.csv','w');
    imageNames = testInfo.paths;
    for ii=1:length(imageNames)
        % remove path from image name
        imName = erase(imageNames{ii},'Test/');
        %imName = imageNames{ii};
        % Format is imageName;Width;Height;X1;Y1;X2;Y2;ClassId
        fprintf(outFid,'%s;%d;%d;%d;%d;%d;%d;%d\n',...
            imName,testInfo.widths(ii),testInfo.heights(ii),...
            testInfo.roiX1(ii),testInfo.roiY1(ii),...
            testInfo.roiX2(ii),testInfo.roiY2(ii),...
            pLabels(ii));
        fprintf(gtFid,'%s;%d;%d;%d;%d;%d;%d;%d\n',...
            imName,testInfo.widths(ii),testInfo.heights(ii),...
            testInfo.roiX1(ii),testInfo.roiY1(ii),...
            testInfo.roiX2(ii),testInfo.roiY2(ii),...
            testInfo.classes(ii));
    end
    fclose(outFid);
    fclose(gtFid);
end