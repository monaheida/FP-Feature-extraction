Feature Extraction in Fingerprint Images
Localization of Diseased Regions

This program requires Python 3.6 or more to be installed, as well as the following libraries:
- numpy (NumPy library)
- cv2 (OpenCV library)
- matplotlib (Matplotlib library)
- skimage (scikit-image library)
- pywt (PyWavelets library)

These commands will allow you to install the required libraries using pip3, enabling you to utilize them in this script:
sudo apt install python3-pip
pip3 install opencv-python
pip3 install scikit-image
pip3 install PyWavelets



Other libraries such as NumPy or Matplotlib should be installed automatically along with aformentioned libraries.



The script is executed using the command: python3.6 bioProject.py <arguments>, where the arguments can be the following:


markfile <img_path> - This argument is used to mark the diseased regions in the fingerprint image. The program will display the marked image, and you can close it by pressing the "q" key on the keyboard.



markfolder <folder_id> - This argument is used to mark the diseased regions for a folder containing multiple images. The <folder_id> parameter can be either 1 or 2, where 1 represents the folder for Eczema images and 2 represents the folder for Verrucas images. The program will process all the images in the specified folder, marking the diseased regions accordingly.




orb <img_path> - This command is used to find and display ORB (Oriented FAST and Rotated BRIEF) features of a fingerprint image.

lbp <img_path> - This command is used to display the Local Binary Pattern (LBP) image and its corresponding histogram.

wavelet <img_path> - This command is used to display the details of an image that has been processed using the Wavelet transform.

sobel <img_path> - This command is used to display an image that has been processed using the Sobel and Laplacian operators.




Two datasets containing images of eczema or verrucas (or whatever disease) will be downloaded to the "./dataset" folder. Additionally, two folders containing examples of processed images with marked diseased regions will be downloaded to the "./markedImg" folder.


The processed results of the "markfolder" command can be found in the "./markedImg/myMarkedImg" directory. Please note that marking fingerprints, especially those with eczema, may take a few minutes to complete.

For the "sobel" and "wavelet" commands, in addition to displaying the processed images, they will also show features from the GLCM (Gray-Level Co-occurrence Matrix) in the console.
