clc;
clear;
exeroot='E:/OneDrive/Projects/densecrf/bin426/examples/Release/dense_inference.exe';
outroot='E:/OneDrive/Projects/TianChiChallenge/data/val_CRF_Output/';
imroot='E:/OneDrive/Projects/TianChiChallenge/data/val_for_CRF/';

notsuccess=[];
parameters=[3.0,3.0,3.0,5.0,5.0,5.0,5.0,3.0,10.0];
%parameters=[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0];
for i=1:1:269
    %[1.34318728659414,2.77316239964340,1.11799568110408,5.69651546736700,5.47506502200618,83.9600893457927,53.6876783910394,23.2294548891047,3.51336354187939];%[2.47484273963235,1.58509686917681,0.431792608504344,6.55347961053601,6.40771446014841,95.8223674524780,34.9900741006371,66.6848270990224,3.08963285947223];
    mkdir([outroot num2str(i-1)]) 
    for j=1:1:48
%          if j~=24
%              continue;
%          end
        %j=24;
        image = imread([imroot 'patch' num2str(i-1) '_' num2str(j-1) '.gif']);
        im=zeros([size(image),3]);
        im(:,:,1)=image;
        im(:,:,2)=image;
        im(:,:,3)=image;
        im=uint8(im);
        
        %imread('../include/densecrf/examples/im3.ppm');
        %unary = -single(image);
        gt=zeros([size(image),3]);
        unary = imread([imroot 'gt' num2str(i-1) '_' num2str(j-1) '.gif'])*255;
        gt(:,:,1)=unary;
        gt(:,:,2)=unary;
        gt(:,:,3)=unary;
        gt=uint8(gt);
        imwrite(im,[outroot 'im.ppm']);
        imwrite(gt,[outroot 'gt.ppm']);
        tic;
        [status result]=system([...
        exeroot ' '...
        [outroot 'im.ppm ']...
        [outroot 'gt.ppm ']...
        [outroot num2str(i-1) '/label' num2str(i-1) '_' num2str(j-1) '.ppm ']...
        [outroot num2str(i-1) '/compare' num2str(i-1) '_' num2str(j-1) '.ppm ']...
        num2str(parameters(1)) ' ' num2str(parameters(2)) ' ' num2str(parameters(3)) ' '... 
        num2str(parameters(4)) ' ' num2str(parameters(5)) ' '... 
        num2str(parameters(6)) ' ' num2str(parameters(7)) ' ' num2str(parameters(8)) ' ' num2str(parameters(9)) ]);    
        toc;
        %label=imread([outroot num2str(i-1) '/label' num2str(i-1) '_' num2str(j-1) '.ppm ']);
        %if(max(max(max(label)))==0)
        %    notsuccess=[notsuccess,i];
        %end
    end
    disp(i);
end