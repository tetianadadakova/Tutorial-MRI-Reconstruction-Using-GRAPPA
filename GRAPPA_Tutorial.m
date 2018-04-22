% Code by Tetiana Dadakova
% 04.21.2018
% tetiana.d@gmail.com

% This is a tutorial on MR image reconstruction using GRAPPA
% Shepp-Logan phantom is first undersampled and then reconstructed using
% GRAPPA

close all

%% 1. Load and display Shepp-Logan phantom

phantom_temp = phantom('Modified Shepp-Logan',64);
% Create complex phantom image
phantom_shl = phantom_temp .* exp(1i * phantom_temp); 

% Display magnitude, phase, real, and imaginary parts of the phantom image 
figure; 
subplot(2,2,1); imshow(abs(phantom_shl),[]); title('abs')
subplot(2,2,2); imshow(angle(phantom_shl),[]); title('angle')
subplot(2,2,3); imshow(real(phantom_shl),[]); title('real')
subplot(2,2,4); imshow(imag(phantom_shl),[]); title('imag')

%% 2. Create phantom images for artificial coil channel sensitivities

% Artificial coil channels sensitivities (6 channels):
% linear gradients in 6 directions
% adding more channels will improve the reconstruction
ramp1 = double(repmat(1:2:128, [64, 1]));
ramp2 = double(repmat(1:2:256, [128, 1]));
ch_sensitivity_1 = ramp1/64; % left-right gradient
ch_sensitivity_2 = imrotate(ch_sensitivity_1, 90); % bottom-up gradient
ch_sensitivity_3 = imrotate(ch_sensitivity_1, 180); % right-left gradient
ch_sensitivity_4 = imrotate(ch_sensitivity_1, 270); % up-bottom gradient
temp = imrotate(ramp2/64, 45);
ch_sensitivity_5 = temp(60:123,60:123); % diagonal left-right gradient
ch_sensitivity_6 = imrotate(ch_sensitivity_5, 180); % diagonal right-left 

% Display channels sensitivities
figure;
subplot(2,3,1);imshow(ch_sensitivity_1,[]); title('Ch 1 sensitivity')
subplot(2,3,2);imshow(ch_sensitivity_2,[]); title('Ch 2 sensitivity')
subplot(2,3,3);imshow(ch_sensitivity_3,[]); title('Ch 3 sensitivity')
subplot(2,3,4);imshow(ch_sensitivity_4,[]); title('Ch 4 sensitivity')
subplot(2,3,5);imshow(ch_sensitivity_5,[]); title('Ch 5 sensitivity')
subplot(2,3,6);imshow(ch_sensitivity_6,[]); title('Ch 6 sensitivity')

% Phantom image accounted for coil channels sensitivities
phantom_ch_1 = phantom_shl .* ch_sensitivity_1;
phantom_ch_2 = phantom_shl .* ch_sensitivity_2;
phantom_ch_3 = phantom_shl .* ch_sensitivity_3;
phantom_ch_4 = phantom_shl .* ch_sensitivity_4;
phantom_ch_5 = phantom_shl .* ch_sensitivity_5;
phantom_ch_6 = phantom_shl .* ch_sensitivity_6;

% Display magnitude
figure;
subplot(2,3,1);imshow(abs(phantom_ch_1),[]);
subplot(2,3,2);imshow(abs(phantom_ch_2),[]);
subplot(2,3,3);imshow(abs(phantom_ch_3),[]);
subplot(2,3,4);imshow(abs(phantom_ch_4),[]);
subplot(2,3,5);imshow(abs(phantom_ch_5),[]);
subplot(2,3,6);imshow(abs(phantom_ch_6),[]);

%% 3. Create undersampled k-space

% 2D Fourier transform image to frequency space
phantom_ch_1_k = fftshift(fft(ifftshift(phantom_ch_1,1),[],1),1);
phantom_ch_1_k = fftshift(fft(ifftshift(phantom_ch_1_k,2),[],2),2);
phantom_ch_2_k = fftshift(fft(ifftshift(phantom_ch_2,1),[],1),1);
phantom_ch_2_k = fftshift(fft(ifftshift(phantom_ch_2_k,2),[],2),2);
phantom_ch_3_k = fftshift(fft(ifftshift(phantom_ch_3,1),[],1),1);
phantom_ch_3_k = fftshift(fft(ifftshift(phantom_ch_3_k,2),[],2),2);
phantom_ch_4_k = fftshift(fft(ifftshift(phantom_ch_4,1),[],1),1);
phantom_ch_4_k = fftshift(fft(ifftshift(phantom_ch_4_k,2),[],2),2);
phantom_ch_5_k = fftshift(fft(ifftshift(phantom_ch_5,1),[],1),1);
phantom_ch_5_k = fftshift(fft(ifftshift(phantom_ch_5_k,2),[],2),2);
phantom_ch_6_k = fftshift(fft(ifftshift(phantom_ch_6,1),[],1),1);
phantom_ch_6_k = fftshift(fft(ifftshift(phantom_ch_6_k,2),[],2),2);

% Display fully-sampled k-space
figure; 
subplot(2,3,1); imshow(abs(phantom_ch_1_k),[1 50]); title('Ch 1 k-space')
subplot(2,3,2); imshow(abs(phantom_ch_2_k),[1 50]); title('Ch 2 k-space')
subplot(2,3,3); imshow(abs(phantom_ch_3_k),[1 50]); title('Ch 3 k-space')
subplot(2,3,4); imshow(abs(phantom_ch_4_k),[1 50]); title('Ch 4 k-space')
subplot(2,3,5); imshow(abs(phantom_ch_5_k),[1 50]); title('Ch 5 k-space')
subplot(2,3,6); imshow(abs(phantom_ch_6_k),[1 50]); title('Ch 6 k-space')

% k-space undersampled twice (each second line is set to zeros)
phantom_ch_1_k_u = zeros(size(phantom_ch_1_k)); 
phantom_ch_1_k_u(1:2:end,:) = phantom_ch_1_k(1:2:end,:);
phantom_ch_2_k_u = zeros(size(phantom_ch_2_k)); 
phantom_ch_2_k_u(1:2:end,:) = phantom_ch_2_k(1:2:end,:);
phantom_ch_3_k_u = zeros(size(phantom_ch_3_k)); 
phantom_ch_3_k_u(1:2:end,:) = phantom_ch_3_k(1:2:end,:);
phantom_ch_4_k_u = zeros(size(phantom_ch_4_k)); 
phantom_ch_4_k_u(1:2:end,:) = phantom_ch_4_k(1:2:end,:);
phantom_ch_5_k_u = zeros(size(phantom_ch_5_k)); 
phantom_ch_5_k_u(1:2:end,:) = phantom_ch_5_k(1:2:end,:);
phantom_ch_6_k_u = zeros(size(phantom_ch_6_k)); 
phantom_ch_6_k_u(1:2:end,:) = phantom_ch_6_k(1:2:end,:);

% Display undersampled k-space
figure;
subplot(2,3,1); imshow(abs(phantom_ch_1_k_u),[1 50]); title('Ch 1 undersampled k-space')
subplot(2,3,2); imshow(abs(phantom_ch_2_k_u),[1 50]); title('Ch 2 undersampled k-space')
subplot(2,3,3); imshow(abs(phantom_ch_3_k_u),[1 50]); title('Ch 3 undersampled k-space')
subplot(2,3,4); imshow(abs(phantom_ch_4_k_u),[1 50]); title('Ch 4 undersampled k-space')
subplot(2,3,5); imshow(abs(phantom_ch_5_k_u),[1 50]); title('Ch 5 undersampled k-space')
subplot(2,3,6); imshow(abs(phantom_ch_6_k_u),[1 50]); title('Ch 6 undersampled k-space')

% FT back to image space, to illustrate the ghosting artifact 
phantom_ch_1_im_u = fftshift(ifft(ifftshift(phantom_ch_1_k_u,1),[],1),1);
phantom_ch_1_im_u = fftshift(ifft(ifftshift(phantom_ch_1_im_u,2),[],2),2);
phantom_ch_2_im_u = fftshift(ifft(ifftshift(phantom_ch_2_k_u,1),[],1),1);
phantom_ch_2_im_u = fftshift(ifft(ifftshift(phantom_ch_2_im_u,2),[],2),2);
phantom_ch_3_im_u = fftshift(ifft(ifftshift(phantom_ch_3_k_u,1),[],1),1);
phantom_ch_3_im_u = fftshift(ifft(ifftshift(phantom_ch_3_im_u,2),[],2),2);
phantom_ch_4_im_u = fftshift(ifft(ifftshift(phantom_ch_4_k_u,1),[],1),1);
phantom_ch_4_im_u = fftshift(ifft(ifftshift(phantom_ch_4_im_u,2),[],2),2);
phantom_ch_5_im_u = fftshift(ifft(ifftshift(phantom_ch_5_k_u,1),[],1),1);
phantom_ch_5_im_u = fftshift(ifft(ifftshift(phantom_ch_5_im_u,2),[],2),2);
phantom_ch_6_im_u = fftshift(ifft(ifftshift(phantom_ch_6_k_u,1),[],1),1);
phantom_ch_6_im_u = fftshift(ifft(ifftshift(phantom_ch_6_im_u,2),[],2),2);

phantom_u_magn = abs(sqrt(phantom_ch_1_im_u.^2 + phantom_ch_2_im_u.^2 + ...
                          phantom_ch_3_im_u.^2 + phantom_ch_4_im_u.^2 + ...
                          phantom_ch_5_im_u.^2 + phantom_ch_6_im_u.^2));
figure;
imshow(imresize(phantom_u_magn,5),[]); 
title({'Magnitude image of phantom', 'from twice undersampled k-space'})

%% 4. Choose autocalibration lines in kspace

% Choose middle 16 lines in k-space for autocalibration
phantom_ch_1_k_acl = zeros(size(phantom_ch_1_k));  
phantom_ch_1_k_acl(25:40,:) = phantom_ch_1_k(25:40,:);
phantom_ch_2_k_acl = zeros(size(phantom_ch_2_k));
phantom_ch_2_k_acl(25:40,:) = phantom_ch_2_k(25:40,:);
phantom_ch_3_k_acl = zeros(size(phantom_ch_3_k));
phantom_ch_3_k_acl(25:40,:) = phantom_ch_3_k(25:40,:);
phantom_ch_4_k_acl = zeros(size(phantom_ch_4_k));
phantom_ch_4_k_acl(25:40,:) = phantom_ch_4_k(25:40,:);
phantom_ch_5_k_acl = zeros(size(phantom_ch_5_k));
phantom_ch_5_k_acl(25:40,:) = phantom_ch_5_k(25:40,:);
phantom_ch_6_k_acl = zeros(size(phantom_ch_6_k));
phantom_ch_6_k_acl(25:40,:) = phantom_ch_6_k(25:40,:);

% Display them
figure;
subplot(2,3,1); imshow(abs(phantom_ch_1_k_acl),[1 50]); title('Ch 1 autocalibration lines')
subplot(2,3,2); imshow(abs(phantom_ch_2_k_acl),[1 50]); title('Ch 2 autocalibration lines')
subplot(2,3,3); imshow(abs(phantom_ch_3_k_acl),[1 50]); title('Ch 3 autocalibration lines')
subplot(2,3,4); imshow(abs(phantom_ch_4_k_acl),[1 50]); title('Ch 4 autocalibration lines')
subplot(2,3,5); imshow(abs(phantom_ch_5_k_acl),[1 50]); title('Ch 5 autocalibration lines')
subplot(2,3,6); imshow(abs(phantom_ch_6_k_acl),[1 50]); title('Ch 6 autocalibration lines')

%% 5. Move kernel of size 3x3 to create the matrix of weights

% Reshape each 3x3 patch in a vector and put all of them into a temp source 
% matrix for each channel of a coil
kNo = 1; % kernel/patch number
for ny = 25:(40-2)
    for nx = 1:(64-2)
        S_ch_1_temp(kNo,:) = ...
            reshape(phantom_ch_1_k_acl(ny:ny+2,nx:nx+2)',[1,9]); % ch1: each patch 3x3 is reshaped into vector and put into matrix one line after another
        S_ch_2_temp(kNo,:) = ...
            reshape(phantom_ch_2_k_acl(ny:ny+2,nx:nx+2)',[1,9]); 
        S_ch_3_temp(kNo,:) = ...
            reshape(phantom_ch_3_k_acl(ny:ny+2,nx:nx+2)',[1,9]); 
        S_ch_4_temp(kNo,:) = ...
            reshape(phantom_ch_4_k_acl(ny:ny+2,nx:nx+2)',[1,9]); 
        S_ch_5_temp(kNo,:) = ...
            reshape(phantom_ch_5_k_acl(ny:ny+2,nx:nx+2)',[1,9]); 
        S_ch_6_temp(kNo,:) = ...
            reshape(phantom_ch_6_k_acl(ny:ny+2,nx:nx+2)',[1,9]); 
        kNo = kNo + 1; % to move through all patches
    end
end
% size(S_ch_1_temp)

% Remove three middle ("unknown") values
% The remaiming values form source matrix S, for each channel
S_ch_1 = S_ch_1_temp(:,[1:3,7:9]);
S_ch_2 = S_ch_2_temp(:,[1:3,7:9]);
S_ch_3 = S_ch_3_temp(:,[1:3,7:9]);
S_ch_4 = S_ch_4_temp(:,[1:3,7:9]);
S_ch_5 = S_ch_5_temp(:,[1:3,7:9]);
S_ch_6 = S_ch_6_temp(:,[1:3,7:9]);

% Middle points form target vector T for each channel
T_ch_1 = S_ch_1_temp(:,5);
T_ch_2 = S_ch_2_temp(:,5);
T_ch_3 = S_ch_3_temp(:,5);
T_ch_4 = S_ch_4_temp(:,5);
T_ch_5 = S_ch_5_temp(:,5);
T_ch_6 = S_ch_6_temp(:,5);

% Stack all the channels together in the same matrix one after another
S = [S_ch_1 S_ch_2 S_ch_3 S_ch_4 S_ch_5 S_ch_6];
T = [T_ch_1 T_ch_2 T_ch_3 T_ch_4 T_ch_5 T_ch_6];

% Invert S to find weights
W = pinv(S) * T;

%% 6. Forward problem to find missing lines

% Construct source matric from the undersampled image
kNo = 1; % kernel/patch number
for ny = 1:2:(64-2)
    for nx = 1:(64-2)
        S_ch_1_new_temp(kNo,:) = ...
            reshape(phantom_ch_1_k_u(ny:ny+2,nx:nx+2)',[1,9]);
        S_ch_2_new_temp(kNo,:) = ...
            reshape(phantom_ch_2_k_u(ny:ny+2,nx:nx+2)',[1,9]);
        S_ch_3_new_temp(kNo,:) = ...
            reshape(phantom_ch_3_k_u(ny:ny+2,nx:nx+2)',[1,9]);
        S_ch_4_new_temp(kNo,:) = ...
            reshape(phantom_ch_4_k_u(ny:ny+2,nx:nx+2)',[1,9]);
        S_ch_5_new_temp(kNo,:) = ...
            reshape(phantom_ch_5_k_u(ny:ny+2,nx:nx+2)',[1,9]);
        S_ch_6_new_temp(kNo,:) = ...
            reshape(phantom_ch_6_k_u(ny:ny+2,nx:nx+2)',[1,9]);
        kNo = kNo + 1;
    end
end
% size(S_ch_1_new_temp)

% Remove three middle ("unknown") values
% The remaiming values form matrix S, for each channel
S_ch_1_new = S_ch_1_new_temp(:,[1:3,7:9]);
S_ch_2_new = S_ch_2_new_temp(:,[1:3,7:9]);
S_ch_3_new = S_ch_3_new_temp(:,[1:3,7:9]);
S_ch_4_new = S_ch_4_new_temp(:,[1:3,7:9]);
S_ch_5_new = S_ch_5_new_temp(:,[1:3,7:9]);
S_ch_6_new = S_ch_6_new_temp(:,[1:3,7:9]);

% Stack all the channels together in the same matrix one after another
S_new = [S_ch_1_new S_ch_2_new S_ch_3_new S_ch_4_new S_ch_5_new S_ch_6_new];

% T_unknown = S_undersampled * W
T_new = S_new * W;

%% 7. Filling in the missing lines into undersampled image

T_ch_1_new_M = reshape(T_new(:,1),[62,31]); % reshape vector into matrix
T_ch_2_new_M = reshape(T_new(:,2),[62,31]);
T_ch_3_new_M = reshape(T_new(:,3),[62,31]);
T_ch_4_new_M = reshape(T_new(:,4),[62,31]);
T_ch_5_new_M = reshape(T_new(:,5),[62,31]);
T_ch_6_new_M = reshape(T_new(:,6),[62,31]);

T_ch_1_new_M = T_ch_1_new_M';
T_ch_2_new_M = T_ch_2_new_M';
T_ch_3_new_M = T_ch_3_new_M';
T_ch_4_new_M = T_ch_4_new_M';
T_ch_5_new_M = T_ch_5_new_M';
T_ch_6_new_M = T_ch_6_new_M';

% Fill in approximated lines
P1_f_u_new = phantom_ch_1_k_u;
P2_f_u_new = phantom_ch_2_k_u;
P3_f_u_new = phantom_ch_3_k_u;
P4_f_u_new = phantom_ch_4_k_u;
P5_f_u_new = phantom_ch_5_k_u;
P6_f_u_new = phantom_ch_6_k_u;
P1_f_u_new(2:2:end-1,2:end-1) = T_ch_1_new_M;
P2_f_u_new(2:2:end-1,2:end-1) = T_ch_2_new_M;
P3_f_u_new(2:2:end-1,2:end-1) = T_ch_3_new_M;
P4_f_u_new(2:2:end-1,2:end-1) = T_ch_4_new_M;
P5_f_u_new(2:2:end-1,2:end-1) = T_ch_5_new_M;
P6_f_u_new(2:2:end-1,2:end-1) = T_ch_6_new_M;

% Display undersampled k-space
figure;
subplot(2,3,1); imshow(abs(phantom_ch_1_k_u),[1 50]); title('Ch 1 undersampled k-space')
subplot(2,3,2); imshow(abs(phantom_ch_2_k_u),[1 50]); title('Ch 2 undersampled k-space')
subplot(2,3,3); imshow(abs(phantom_ch_3_k_u),[1 50]); title('Ch 3 undersampled k-space')
subplot(2,3,4); imshow(abs(phantom_ch_4_k_u),[1 50]); title('Ch 4 undersampled k-space')
subplot(2,3,5); imshow(abs(phantom_ch_5_k_u),[1 50]); title('Ch 5 undersampled k-space')
subplot(2,3,6); imshow(abs(phantom_ch_6_k_u),[1 50]); title('Ch 6 undersampled k-space')

% Display k-space after the approximated lines were filled in
figure;
subplot(2,3,1); imshow(abs(P1_f_u_new),[1 50]); title('Ch 1 restored k-space')
subplot(2,3,2); imshow(abs(P2_f_u_new),[1 50]); title('Ch 2 restored k-space')
subplot(2,3,3); imshow(abs(P3_f_u_new),[1 50]); title('Ch 3 restored k-space')
subplot(2,3,4); imshow(abs(P4_f_u_new),[1 50]); title('Ch 4 restored k-space')
subplot(2,3,5); imshow(abs(P5_f_u_new),[1 50]); title('Ch 5 restored k-space')
subplot(2,3,6); imshow(abs(P6_f_u_new),[1 50]); title('Ch 6 restored k-space')

% Fourier transform restored k-space for each channel
Im_Recon_ch_1 = fftshift(ifft(ifftshift(P1_f_u_new,1),[],1),1);
Im_Recon_ch_1 = fftshift(ifft(ifftshift(Im_Recon_ch_1,2),[],2),2);
Im_Recon_ch_2 = fftshift(ifft(ifftshift(P2_f_u_new,1),[],1),1);
Im_Recon_ch_2 = fftshift(ifft(ifftshift(Im_Recon_ch_2,2),[],2),2);
Im_Recon_ch_3 = fftshift(ifft(ifftshift(P3_f_u_new,1),[],1),1);
Im_Recon_ch_3 = fftshift(ifft(ifftshift(Im_Recon_ch_3,2),[],2),2);
Im_Recon_ch_4 = fftshift(ifft(ifftshift(P4_f_u_new,1),[],1),1);
Im_Recon_ch_4 = fftshift(ifft(ifftshift(Im_Recon_ch_4,2),[],2),2);
Im_Recon_ch_5 = fftshift(ifft(ifftshift(P5_f_u_new,1),[],1),1);
Im_Recon_ch_5 = fftshift(ifft(ifftshift(Im_Recon_ch_5,2),[],2),2);
Im_Recon_ch_6 = fftshift(ifft(ifftshift(P6_f_u_new,1),[],1),1);
Im_Recon_ch_6 = fftshift(ifft(ifftshift(Im_Recon_ch_6,2),[],2),2);

% Display restores images for each channel
figure;
subplot(2,3,1); imshow(abs(Im_Recon_ch_1),[]); title('Reconstructed image, ch 1')
subplot(2,3,2); imshow(abs(Im_Recon_ch_2),[]); title('Reconstructed image, ch 2')
subplot(2,3,3); imshow(abs(Im_Recon_ch_3),[]); title('Reconstructed image, ch 3')
subplot(2,3,4); imshow(abs(Im_Recon_ch_4),[]); title('Reconstructed image, ch 4')
subplot(2,3,5); imshow(abs(Im_Recon_ch_5),[]); title('Reconstructed image, ch 5')
subplot(2,3,6); imshow(abs(Im_Recon_ch_6),[]); title('Reconstructed image, ch 6')

% Combine coil channels
Im_Recon = sqrt(abs(Im_Recon_ch_1).^2 + abs(Im_Recon_ch_2).^2 + ...
                abs(Im_Recon_ch_3).^2 + abs(Im_Recon_ch_4).^2 + ...
                abs(Im_Recon_ch_5).^2 + abs(Im_Recon_ch_6).^2);

% Display image reconstructed uisng GRAPPA
figure; imshow(imresize(Im_Recon,5),[]); title('Image reconstructed using GRAPPA')




