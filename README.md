# Tutorial-MRI-Reconstruction-Using-GRAPPA
I wrote this code when trying to understand how GRAPPA reconstruction works. I was trying to keep it very simple and readable. Thought, it might be useful for someone else. Let me know if you have questions, comments, or suggestions: tetiana.d@gmail.com


Code consists of the following parts:

1. Load Shepp-Logan phantom.
![alt text](fig/01_SLphantom.png)

2. Create six artificial sensitivities of the channels of the coil. This is done by creating linear intensity gradients in 6 directions. Adding more channels will improve the reconstruction.
![alt text](fig/02_CoilChSensitivities.png)
![alt text](fig/03_SLphantomByCh.png)

3. Fourier transform each channel image to k-space
![alt text](fig/04_kSpaceFullySampled.png)
and undersample it twice.
![alt text](fig/05_kSpaceUnderSampled.png)
Reconstruct the Shepp-Logan phantom image from the under sampled k-space to show the artifact.
![alt text](fig/06_SLphantomAliased.png)

4. Choose auto calibration lines.
![alt text](fig/07_AutocalibrationLines.png)

5. Source (S) pixels are related to target (T) pixels by weighs (W): S * W = T. To find weights: W = inv(S) * T. (See schematic figure here: http://mriquestions.com/grappaarc.html)

6. Forward problem to find missing lines. T_unknown = S_undersampled * W

7. Fill in the missing lines into undersampled image.
![alt text](fig/08_kSpaceRestored.png)
Reconstruct image for each channel.
![alt text](fig/09_ReconstructedImageByChannel.png)
And finally reconstruct the image.
![alt text](fig/10_ReconstructedImage.png)




