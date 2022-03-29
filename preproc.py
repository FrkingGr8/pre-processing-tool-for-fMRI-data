#!/usr/bin/python3
import nibabel as nib
import numpy as np
from scipy.fftpack import fft, ifft
import argparse
import sys


# The class has two variables input and output.
# Input is the NIFTI image not the input file name
# Output is the name of the output file.
class PreProc:

    # Intializer function for class PreProc. This assigns all the values needed for Pre Processing the image.
    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.image = nib.load(self.input)
        self.header = self.image.header
        self.TR = self.header.get_zooms()[3]*1000
        self.voxeldim = self.header.get_zooms()[:3]
        self.fMRIdata = self.image.get_fdata()
        self.x , self.y, self.z, self.v = self.image.shape


#------------------------------------------------------------------------------------------------------------------       
# Utility functions used to save the result and show the value of different parameters of class
    # Prints all the relevant values stored in the object of class PreProc 
    def print_values(self):
        print('shape of fMRI data file is: ', self.fMRIdata.shape)
        print('dim in x axis is: ', self.x)
        print('dim in y axis is: ', self.y)
        print('dim in z axis is: ', self.z)
        print('dim in volume axis is: ', self.v)
        print('value of TR in miliseconds is: ', self.TR)
        print('Dimension of voxel in mm is: ', self.voxeldim)
        print('Result will be stored in the file with prefix: ', self.output)
    

    # Function to save the fMRI data of the object in a file with extension .nii.gz
    def savefMRI(self):
        outputImage = nib.Nifti1Image(self.fMRIdata, np.eye(4), header=self.header)
        outputextension = '.nii.gz'
        outputfile = self.output + outputextension
        nib.save(outputImage, outputfile)


#------------------------------------------------------------------------------------------------------------------
# These two function perofrms Slice Time Correction on full fMRI data
    # Performs Linear Interpolation on a single time series volume
    def LinearInterpolation(self, x, y, index, sAcqTime, targetTime):
        # Get the single time series from the whole fMRI data
        svolume = self.fMRIdata[x][y][index]
        ycorrected = svolume
        y1 = svolume[:-1]
        y2 = svolume[1:]

        # Setting a correct coefficient for linear interpolation 
        coefficient = (targetTime - sAcqTime)
        if(coefficient < 0):
            coefficient =  coefficient + self.TR
        coefficient = coefficient/self.TR

        if (targetTime < sAcqTime):
            y1 = np.concatenate(([2*svolume[0]-svolume[1]],svolume[:-2]))
            y2 = svolume[:-1]        

        # calculating correct value of the time series using the formula y = y1 + coefficient*(y2-y1)
        ycorrected = np.concatenate((y1 + coefficient*(y2-y1), [svolume[-1]])) 
            

        return ycorrected


    # This fucntion gives individual time series volume data to LinearInterpolation function to calculate Slice Time Correction on full image
    def sTCorrection(self, targetTime, sliceTimeAcqFile):
        # Opening the image in appending format
        outputfile = open(self.output + '.txt', 'a')
        sfile = np.loadtxt(sliceTimeAcqFile)

        # First check condition on target time wrt TR
        if(self.TR < targetTime or targetTime < 0):
            outputfile.write("SLICE TIME CORRECTION FAILURE\n")
        else:
            finalImage = np.zeros(np.shape(self.fMRIdata))
            
            for index in range(self.z):
                if(sfile[index]>self.TR or sfile[index]<0):
                    outputfile.write("SLICE TIME CORRECTION FAILURE\n")
                    break

                for i in range(self.x):
                    for j in range(self.y):
                        finalImage[i][j][index] = self.LinearInterpolation(i, j, index, sfile[index], targetTime)

            outputfile.write("SLICE TIME CORRECTION SUCCESS\n")
            self.fMRIdata = finalImage



#------------------------------------------------------------------------------------------------------------------
# These two function performs Temporal Filtering on the full fMRI data
    #Performing Temporal filtering on single voxel volume
    def TFVoxel(self, voxelTimeSeries, hfreq, lfreq):
        vTimeSeries = voxelTimeSeries
        nSamplePoints = np.shape(voxelTimeSeries)[0]
        frequencies = np.concatenate( (np.linspace(0.0, 1000.0/(2.0*self.TR), nSamplePoints//2),  np.linspace(-1000.0/(2.0*self.TR),0.0, nSamplePoints//2)))
        yFourier = fft(vTimeSeries)
        yFourier[np.abs(frequencies) > hfreq] = 0
        yFourier[np.abs(frequencies) < lfreq] = 0
        finalTimeSeries = ifft(yFourier)
        # print(type(finalTimeSeries))
        return finalTimeSeries


    # This function gives individual voxel volume to the method TFVoxel. Doing this gives us the temporal filtering on whole image
    def temporalFiltering(self, high, low):
        hfreq = 1/low
        lfreq = 1/high

        # Making sure the hfreq >= lfreq
        if hfreq < lfreq:
            temp = hfreq
            hfreq = lfreq
            lfreq = temp

        finalImage = np.zeros(np.shape(self.fMRIdata))
        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    finalImage[i][j][k] = self.TFVoxel(self.fMRIdata[i][j][k], hfreq, lfreq)

        self.fMRIdata = finalImage



#------------------------------------------------------------------------------------------------------------------
# These three function performs spatial smoothing on the full fMRI data
    # make gaussian kernel using voxelwidth, fwhm and kernel size
    def gaussianKernel(self, ksize, fwhm, voxelwidth):
        x = np.arange(int(-1*ksize/2), int(ksize/2)+1)
        sigma = fwhm/(2.3548*voxelwidth)            # since np.sqrt(8*np.log(2)) = 2.3548
        y = np.exp((-1*x*x)/(2*sigma*sigma))
        kernel = y/sum(y)

        return kernel


    # Apply gaussian filter on a given dimension of the fMRI data
    def applyKernel(self, seq, kernel):
        kLength = np.size(kernel)
        sLength = np.size(seq)

        #Padding is needed so that the dimension of output sequence same as input sequence
        pLength = kLength // 2
        pSequence = np.concatenate((np.zeros(pLength), seq, np.zeros(pLength)))


        # Perform convolution on padded sequence to get the output sequence
        output = np.zeros(sLength)
        for startingPointer in range(0, sLength):
            output[startingPointer] = np.sum(np.multiply(kernel, pSequence[startingPointer:startingPointer+kLength]))

        return output

    
    # This function applies spatial smoothing to a particular axis 0, 1, 2 meaning x, y and z axis respectively
    def smoothAxis(self,timestamp, ksize, fwhm, axis):
        output = np.zeros(np.shape(self.fMRIdata)[:-1])
        if(axis == 0):
            # x axis
            kernel = self.gaussianKernel(ksize=ksize, fwhm=fwhm, voxelwidth=self.voxeldim[0])
            for i in range(self.y):
                for j in range(self.z):
                    output[:,i,j] = self.applyKernel(self.fMRIdata[:,i,j,timestamp],kernel=kernel)

        elif axis == 1:
            # y axis
            kernel = self.gaussianKernel(ksize=ksize, fwhm=fwhm, voxelwidth=self.voxeldim[1])
            for i in range(self.x):
                for j in range(self.z):
                    output[i,:,j] = self.applyKernel(self.fMRIdata[i,:,j,timestamp],kernel=kernel)

            else:
                # z axis
                kernel = self.gaussianKernel(ksize=ksize, fwhm=fwhm, voxelwidth=self.voxeldim[2])
                for i in range(self.x):
                    for j in range(self.y):
                        output[i,j,:] = self.applyKernel(self.fMRIdata[i,j,:,timestamp],kernel=kernel)

        return output


    # This function gives individual sequence to the method applyKernel. This is repeated for each axis that is x, y and z axis. This results to applying spatial smoothing on the image
    def spatialSmoothing(self, fwhm, ksize):
        finalImage = np.zeros(np.shape(self.fMRIdata))

        # We will iterate through each timestamp and perform spatial smooting using 1D kernel
        for tstamp in range(self.v):
            # print("Time Stamp running: ", tstamp)
            # applying kernel smoothing in x axis
            outputX = self.smoothAxis(timestamp=tstamp, ksize=ksize, fwhm=fwhm, axis=0)

            # applying kernel smoothing in y axis
            outputY = self.smoothAxis(timestamp=tstamp, ksize=ksize, fwhm=fwhm, axis=1)

            # applying kernel smoothing in z axis 
            outputZ = self.smoothAxis(timestamp=tstamp, ksize=ksize, fwhm=fwhm, axis=2)


            meanOutput = (outputX + outputY + outputZ)/3
            finalImage[:,:,:,tstamp] = meanOutput
        
        self.fMRIdata = finalImage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputFile", required=True)
    parser.add_argument('-o', '--outputFile', required=True)
    parser.add_argument('-tc', '--listSliceTimeCorrection', nargs='+', default=[0, 0])
    parser.add_argument('-tf', '--listTemporalFiltering', nargs='+', default=[0, 0])
    parser.add_argument('-sm', '--fwhm', default=0)
    args = parser.parse_args()
    
    inputfile = str(args.inputFile)
    outputfile = str(args.outputFile)

    # Make an object of class PreProc
    final = PreProc(input=inputfile, output=outputfile)

    # Take arguments of all the case
    # Arguments of Slice Time Correction
    targetTime = float(args.listSliceTimeCorrection[0])
    sliceTimeAcquisitionFile = args.listSliceTimeCorrection[1]

    # Arguments of Temporal filtering
    high = float(args.listTemporalFiltering[0])
    low = float(args.listTemporalFiltering[1])

    #Argument of Spatial Smoothing default kernel size in 5.
    fwhm = float(args.fwhm)
    ks = 5

    nArguments = len(sys.argv)
    for index in range(nArguments):
        arg = sys.argv[index]

        if arg == '-tc' or arg == '--listTimeSeriesGeneration':
            final.sTCorrection(targetTime=targetTime, sliceTimeAcqFile=sliceTimeAcquisitionFile)
            
        elif arg == '-tf' or arg == '--listTemporalFiltering':
            final.temporalFiltering(high=high, low=low)
        
        elif arg == '-sm' or arg == '--fwhm':
            final.spatialSmoothing(fwhm=fwhm, ksize=ks)

    final.savefMRI()

if __name__=='__main__':
    main()









# References for the assignment are given below:
    # Spatial Smoothing
        # https://users.fmrib.ox.ac.uk/~stuart/thesis/chapter_6/section6_2.html
        # https://support.brainvoyager.com/brainvoyager/functional-analysis-preparation/29-pre-processing/86-spatial-smoothing

    # Temporal Filtering
        # https://lukas-snoek.com/NI-edu/fMRI-introduction/week_4/temporal_preprocessing.html

    # Slice Time correction 
        # https://matthew-brett.github.io/teaching/slice_timing.html
        
        # https://github.com/neurospin/pypreprocess
