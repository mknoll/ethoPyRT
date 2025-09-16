# Installation 

The package can be installed via pip:

    pip install ethoPyRT


# General remarks

This package helps analyze DICOM data exported from the Varian ETHOS system.
The Ethos treatment interface allows per-session export of data, i.e. 

<dcmpath>/<patname>/Session Export/<PID>/<TreatmentIntent>/Session_<n>

where dcmpath is manually selected, <patname> is manually created and the 
subfolders are automatically created during export.



# Usage

## General 

Load the package with 

    import ethoPyRT

The ethoPyRT package provides the RTMetrics class, which is instantiated per patient using its ID pid (correspoding to <PID> above):

    rtm = RTMetrics(dcmpath, basepath, pid, analysisDir)

dcmpath is the path where the exported data from Ethos is stored.
basepath is a folder to store transformed intermediate data generated with ethoPyRT (i.e. nifti files, see below) and 
can be freely selected, as well as as <analysisDir>, where results are stored.


## QC & Preprocessing

First, completeness of data is checked via 

    rtm.checkFiles() 

File conversions to help manual inspection of data is performed with

    rtm.createNifti()

and RT structures are extracted with 

    rtm.extractRTStruct()

Data is stored in 

    <basepath>/<TreatmentIntent>/<PID>/Session_<n>/

in the nifti/ and RTSTRUCT/ subfolders.


## Data loading

RT doses are loaded with 
    
    rtm.loadDoses()

and plans are assigned (scheduled / adapted) with

    rtm.assignPlanType()

RTStructures are loaded with 
    
    rtm.loadMasks()


## Data analysis

DVHs are calculated with 
    
    rtm.calcDVH()

Scheduled vs adapted vs treated plans are compared using 
    
    rtm.compare()

Difference DVHs are computed with 
    
    rtm.diffDVH()

and plotted with 

    rtm.plotDVH()

Dose metrics are calculated with 

    rtm.calcMetrics()

Radiomics features are computed with 

    rtm.calcRadiomics()

The number of adapted / scheduled plans. 

    rtm.countAdpSched()

Registration of CBCTs (currently only tested for prostate)

    rtm.registerCBCT()

and application:
    
    rtm.applyTransform(type="CBCT", ref="CBCT")







