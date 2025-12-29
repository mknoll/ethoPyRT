import logging
import seaborn
import glob
from pydicom import dcmread
import numpy as np
import os 
import shutil
import re
import logging
import seaborn as sns
from matplotlib.colors import ListedColormap 
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from ..calcMetrics.dvh import *
from ..extractRTStruct.extract import *
import pathlib

            
def generate_colormap(N):
    arr = np.arange(N)/N
    N_up = int(math.ceil(N/7)*7)
    arr.resize(N_up)
    arr = arr.reshape(7,N_up//7).T.reshape(-1)
    ret = matplotlib.cm.hsv(arr)
    n = ret[:,3].size
    a = n//2
    b = n-a
    for i in range(3):
        ret[0:n//2,i] *= np.arange(0.2,1,0.8/a)
    ret[n//2:,3] *= np.arange(1,0.1,-0.9/b)
    return ret

class RTMetrics:
    ##outputfiles
    statusDir = "./"
    analysisDir = "./"
    
    ##### RTPLAN + RTDOSE
    tmsS = {} #stores rdose types
    doses = [] #dose files
    dosesFactor = [] #scaling factor to yield Gy values
    headers = [] #headers of dose files
    type = []  # type of dose files

    ##### MASKS
    masksToLoad = None
    masks = []
    added = []

    ##### DVHs
    sortedData = []
    sortedDataSess = []
    ii = []

    #### log debug & info
    log = logging.getLogger('user')
    debug = logging.getLogger('debug')

    ### compare plans
    frames = []
    metrics = []
    names = []
    targets = []

    ##prescription
    totalDose = None
    fractions = None
    prescriptionTarget = None
    intent = None ## treatment intent

    ### adapted or scheduled?
    adapted = 0
    scheduled = 0

    ### conebeamcts
    cbcts = {}
    allCBCT = None
    ### synthCts
    scts = {}
    ### doses
    doseAdp = {}
    doseSched = {}
    doseTreat = {}

    # matching of names
    match = {}

    ############
    ### MRI data
    mriOutpath = "./"
    mris = {}


    ####################
    #####################

    ###################################
    # add info about mri data
    def addMRInfo(self, outpath):
        self.mriOutpath = outpath

    ### Create Nifti from MRI DICOMS ###
    def createNiftiMRI(self, inpath):
        for f2 in glob.glob(inpath + "/*/*/*"):
            ##date 
            dt = f2.split("/")[-1]
            ##get dicom paths
            dcmfl = glob.glob(f2+"/*/*")
            #seq = [x.split("Ser")[1].split(".")[0] for x in dcmfl]
            #img = [x.split("Img")[1].split(".")[0] for x in dcmfl]
            #print(dcmfl)
            #print(f2)
            fldPidDate = self.mriOutpath + "/" + self.pid + "/" + dt + "/nifti/"
            print(fldPidDate)
            os.system("mkdir -p " + fldPidDate)

            ###  get single dicom for each sequence
            cmd = "dcm2niix -f '%p_%e_%s_%j' -o '" + fldPidDate + "' '" + dcmfl[0] + "'"
            dir = os.listdir(fldPidDate)
            if len(dir) == 0:
                print(cmd)
                os.system(cmd)

    ### identify mr sequences ########## ### TODO adjust string for selection
    def identifyMRI(self, typeSel="t2_*tra*.nii", ty="t2"):
        #dates 
        mris = {}
        #for fl in glob.glob("../data_proc_mri/20250223/Prostate/"+ self.pid + "/*"):
        for fl in glob.glob(self.mriOutpath + "/"+ self.pid + "/*"):
            print(fl)
            flD = glob.glob(fl+"/nifti/"+typeSel)[0] # FIXME: können mehrere sein !
            print(flD)
            dt = fl.split("/")[-1]
            mris.update({dt:flD})
        mris = dict(sorted(mris.items()))
        
        #### update global mris
        if ty in self.mris.keys():
            self.mris.update({ty:mris})
        else:
            self.mris[ty] = mris


    ## register MRI to first session CBCT
    def registerMRI(self):
        if not "Session_1" in self.cbcts.keys():
            raise Exception("CBCT not found!") ##FIXME
         
        for ty in self.mris:
            print("------> " + ty)
            mris = self.mris[ty]
            #print(mris)
            for dt in mris:
                #print(dt)
                ### fixed - > first session CBCTs
                fixed = self.cbcts['Session_1'][0]
                moving = mris[dt]

                ### tfransform - create folder
                tfmFolder = moving.split("nifti/")[0] + "/tfm/"
                os.system("mkdir -p " + tfmFolder)
                ## transform filename
                tfmFile = tfmFolder + "mri__"+ moving.split("/")[-1].replace(".nii", "") + "__to__cbct_Session_1.tfm"
                print(tfmFile)
                print("moving: " + moving)
                if not os.path.isfile(tfmFile):
                    self.register(fixed, moving, tfmFile)


    ## apply registration to MRI
    #def applyRegistrationMRI(self):
    def applyRegistrationToMRI(self, ty="t2", defPixelVal=0):
        mris = self.mris[ty]

        ##fixed - CBCT
        fixed = sitk.ReadImage( self.cbcts['Session_1'][0] )
        outFiles = {}
        for dt in mris:
            #print(dt)
            moving = mris[dt]
            tfm = moving.split("nifti/")
            tfmFile = glob.glob(tfm[0] + "tfm/mri__"+tfm[1].replace(".nii", "")+"__to__cbct_Session_1.tfm")[0]
            tfmFile = sitk.ReadTransform(tfmFile)
            print(moving)

            ###            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(defPixelVal) ## images: -1000; dose: 0 ### FIXME!!!
            resampler.SetTransform(tfmFile)
            out = resampler.Execute(sitk.ReadImage(moving))
            #sitk.WriteImage(out, "/tmp/out.nii.gz"

            outFiles.update({dt:out})
        return(outFiles)

    ##########


    def __init__(self, dcmpath, basepath, pid, statusDir="../status/", analysisDir="../analysis/", allCBCT=False, intent="*"):
        ###TODO: check if files exists / pat ID exists 
        self.dcmpath = dcmpath
        self.basepath = basepath 
        self.pid = pid
        self.statusDir = statusDir
        self.analysisDir = analysisDir
        self.allCBCT = allCBCT
        self.intent = intent
        self.debug.setLevel(logging.CRITICAL)
        self.log.setLevel(logging.INFO)

    def __str__(self):
        return f"{self.pid}"

    
    def loadDoses(self):
        """
        Loads the RTDOSE information and checks if all three expected doses 
        are available (ADAPTED_FROM, REFERENCE_PLAN, TREATED_PLAN)
        """
        
        self.log.info("Loading doses...")
        dcmpath = self.dcmpath
        pid = self.pid

        tmsS = {}
        for session in glob.glob(dcmpath + "/*/Session Export/"+ pid + "/"+self.intent+"/*"):
            print(".", end="")
            self.debug.info("SESSION: " + session)
            sess = session.split("/")[8]
            l = {}
            for dcm in glob.glob(dcmpath + "/*/Session Export/"+ pid + "/"+self.intent+"/" + sess + "/RTDO*"):
                session = dcm.split("/")[8]
                self.debug.info("file: " + dcm)
                try:
                    ds = dcmread(dcm) ##FIXME; remove force=True
                    uidose =  str(ds[0x300c, 0x0002][0][0x008, 0x1155]).split("UI: ")[1]
                    self.debug.info("UI: " + uidose)
        
                    tmp = dcm.split("/")
                    tmp.pop()
                    
                    flD = ("/").join(tmp) + "/RTPLAN."+uidose+".dcm"
                    dose = dcmread(flD)

                    cmpSID = str(ds[0x0020,0x000e]).split("UI: ")[1]
                    self.debug.info("cmpSID: " + cmpSID)
                    curTM = str(dose[0x300a, 0x0002]).split("SH: ")[1].replace("'", "")
                    self.debug.info("curTM: " + curTM)
    
                    ##treated plan?
                    treated = str(dose[0x300c, 0x0002][0][0x300a, 0x0055]).split("CS: ")[1].replace("'", "")
                    self.debug.info("treated: " + treated)

                    ###(3004, 000e) Dose Grid Scaling                   DS: '0.00197119654762'
                    scalingFactor = str(ds[0x3004,0x000e]).split("DS: ")[1].replace("'", "")
                    self.debug.info("scalingFactor: " + scalingFactor)
                    l.update({cmpSID:[curTM, treated, scalingFactor]})
                    

                    ###added
                    iPTV = int(str(dose[0x300a, 0x0010]).split("length ")[1].replace(">", ""))
                    for ip in range(iPTV):
                        trgIp = str(dose[0x300a, 0x0010][ip][0x300a,0x0016]).split("LO: ")[1].replace("'", "")
                        if trgIp == "PTV":
                            self.prescriptionTarget = trgIp ###FIXME!!!!!
                        totalDose = str(dose[0x300a, 0x0010][ip][0x300a,0x0026]).split("DS: ")[1].replace("'", "")
                        self.totalDose = totalDose
                    nFract = str(dose[0x300a, 0x0070][0][0x300a, 0x0078]).split("IS: ")[1].replace("\'", "")
                    self.fractions = nFract
                    
                except Exception as e:
                    self.debug.warning("ERROR with " + dcm)
                    self.debug.warning(e)
            tmsS.update({sess:l})
        self.tmsS = tmsS

        ### check if 3 IDs / session were identified
        for t in tmsS:
            if len(tmsS[t]) != 3:
                self.debug.critical("Nicht 3 IDs/Pläne pro Session verhanden")
                self.debug.critical(tmsS[t])
                self.log.critical("Nicht 3 IDs/Pläne pro Session verhanden")
                raise ValueError('Check Data!')

        ## ggf check prescription target, TD, ED

    
    def assignPlanType(self):
        """
        Extract the type of RTPLAN
        """
        
        self.log.info("Determine RTPLAN type ...")

        basepath = glob.glob(self.basepath + "/"+self.intent+"/"+self.pid+ "/")[0] ##FIXME
        tmsS = self.tmsS
        
        ## TODO: check len of basepath
        doses = []
        dosesFactor = []
        headers = []
        type = []
        
        for files in glob.glob(basepath):
            for sess in glob.glob(files+"/*"):
                self.debug.info(sess)
                for name0 in glob.glob(sess + "/nifti/*.nii"):
                    self.debug.info("name0: " + name0)
                    name = name0.split("/")[-1] 
                    self.debug.info("name new: " + name0)
                    root = ("/").join(name0.split("/")[:-1])
                    self.debug.info("root new: " + root)
                    if  not "sct_" in name:
                        tmp = str(name).split("_")
                        if len(name.split("_")[2]) > 3:
                            n = nib.load(str(root)+"/"+name)
                            if n.header['pixdim'][1] > 1:
                                headers.append(n)
                                pid = str(root).split("/")[-3] ## check with self
                                self.debug.info("pid new: " + pid)
                                session = str(root).split("/")[-2]
                                self.debug.info("session new: " + session)
                                organ = str(root).split("/")[-4]
                                self.debug.info("organ new: " + organ)
                                sid = name.split("_")[3].replace(".nii", "")
                                try:
                                    tms = tmsS[session]
                                    indTM= {k:i for i,k in enumerate(tms.keys())}
                                    indTM
                                    try:
                                        ty = "unknown"
                                        if "ADP" in tms[sid][0]:
                                            ty = "adapted"
                                        if "SCH" in tms[sid][0]:
                                            ty = "scheduled"
                                        type.append(ty+":"+tms[sid][1]+":"+ tms[sid][0])
                                        doses.append(str(root)+"/"+name)
                                        dosesFactor.append(tms[sid][2])
                                    except Exception as e: 
                                        self.debug.warning("ERROR")
                                        self.debug.warning(e)
                                except Exception as e :
                                    self.debug.warning(e)
        #self.doses = doses
        self.doses = [x.replace("//", "/") for x in doses] ### FIXME
        self.dosesFactor = dosesFactor
        self.headers = headers
        self.type = type

    
    def loadMasks(self, masksToLoad=None):
        """
        Load masks 
        """
        self.log.info("Loading masks...")
        ### get exported masks / check # of masks
        masks = []
        added = []
        doses = self.doses
        
        for i in range(len(self.doses)):
            self.debug.info(i)
            print(str(i+1) + "/" + str(len(self.doses)), end="\r")
            tmp = doses[i].split("/")
            session = [x for x in tmp if x.startswith("Session_") and not ".nii" in x]
            if session not in added: 
                self.debug.info(session)
                added.append(session)
                path = ("/").join(tmp[0:5])+"/" + session[0] + "/RTSTRUCT/mask*"
                labels = {}
                for file in glob.glob(path):
                    #print(file)
                    #nm = file.split("_")[4].split(".")[0]
                    nm = file.split("mask_")[1].split(".")[0]
                    if masksToLoad is not None:
                        if not nm in masksToLoad:
                            continue
                        #print(nm)
                    try:
                        labels.update({nm:sitk.ReadImage(file)})
                    except Exception as e: 
                        self.debug.error("Could not read: " + file)
                        self.debug.error(e)
                masks.append(labels)
    
        self.masks= masks
        self.added = added

        ### check if masks were added
        for m in masks:
            if len(m) == 0:
                self.debug.critical("No masks added!")
                self.log.critical("No masks added!")
                raise
                
            

        
    def calcDVH(self, normalize=False):      
        """
        Calculate DVHs 
        
            Parameters:
                normalize (bool): Divide the ADAPTED_FROM and REFERENCE_PLAN dose information by fractions
                
        """
        self.log.info("Calculate DVHs...")
        
        ##https://github.com/pyplati/platipy/blob/master/platipy/imaging/dose/dvh.py
        added = self.added
        doses = self.doses
        dosesFactor = self.dosesFactor
        masks = self.masks
        type = self.type
        
        data = []
        dataSess = []
        for i in range(len(doses)):
            print(str(i+1)+"/"+str(len(doses))+ "   ", end="\r")
            tmp = doses[i].split("/")
            session = [x for x in tmp if x.startswith("Session_") and not ".nii" in x]
            w = np.where(np.array(added) == session)[0][0]
            if added[w] == session:
                labels = masks[w]
                dose_grid = sitk.ReadImage(doses[i])
                try:
                    dvh = calculate_dvh_for_labels(dose_grid,  labels, bin_width=10)
                    self.debug.info("Dosis factor: " + str(float(dosesFactor[i])))

                    ## normalize to fraction
                    fract = int(self.fractions)
                    ed = float(self.totalDose)/fract

                    factor = 1
                    if normalize and not "TREATED" in type[i]:
                        factor = 1/(fract)

                    dvh['mean'] = dvh['mean']*(factor*float(dosesFactor[i]) *10000) ### ACHTUNG!!! Faktor 10000
                    dvh.columns = np.append(dvh.columns.values[0:3], dvh.columns.values[3:]*(factor*float(dosesFactor[i]) *10000)) ### ACHTUNG!!! Faktor 10000
                    
                    ### scale!
                    if False:
                        fract = int(self.fractions)
                        ed = float(self.totalDose)/fract
                                       
                    data.append(dvh)
                    dataSess.append([session, type[i], doses[i], dosesFactor[i]])
                except Exception as e:
                    self.debug.error(e)

        #### SORTING
        sortedData = []
        sortedDataSess = []
        try:
            ## order data by dataSess
            toSort = pd.DataFrame(dataSess)
            toSort[0] = [x[0] for x in toSort[0]]
            toSort[3] = [x.split("_")[1] for x in toSort[0]]
            toSort.columns = ["Session", "Type", "Dose", "SessionID"]
            toSort['SessionID'] = pd.to_numeric(toSort['SessionID'])
            toSort = toSort.sort_values(by=["SessionID", "Type"])
            ww = toSort.index.values
            

            for i in range(len(ww)):
                sortedData.append(data[ww[i]])
                sortedDataSess.append(dataSess[ww[i]])

            self.sortedData = sortedData
            self.sortedDataSess = sortedDataSess
        except Exception as e:
            self.debug.error(e)


        ### aggregate sessions            
        lastSess = sortedDataSess[0][0][0]
        
        ii = []
        tmp = []
        for i in range(len(sortedDataSess)):
            session = sortedDataSess[i][0][0]
            if session == lastSess:
                lastSess = session
                tmp.append(i)
            else:
                ii.append([tmp, lastSess])
                tmp = []
                lastSess = session
                tmp.append(i)
        ii.append([tmp, lastSess])
        
        self.ii = ii


    def diffDVH(self):
        frames = self.frames
        metrics = self.metrics
        names = self.names
        targets = self.targets
        
        for w in range(len(set(targets))):
            curTar = list(set(targets))[w]
            print(curTar)
            sub = []
            subNM = []
            for r in range(len(targets)):
                if targets[r] == curTar:
                    sub.append(frames[r])
                    subNM.append(names[r])
            
            print("####")
            print(len(sub))
        
            xx = [pd.DataFrame(x['TREATED-REF']).T for x in sub]
            dff = pd.concat(xx,  ignore_index = True)
            dff.index = [x for x in subNM]
            xx = [pd.DataFrame(x['ADAPTED-SCHEDULED']).T for x in sub]
            dff2 = pd.concat(xx,  ignore_index = True)
            dff2.index = [x for x in subNM]
            #%matplotlib inline

            #
            self.debug.info("Write file: " +  self.analysisDir+"/" + self.pid + "__treated_reference.csv")
            dff.to_csv(self.analysisDir+"/" + self.pid + "_" + curTar + "__treated_reference.csv")
            self.debug.info("Write file: " +  self.analysisDir+"/" + self.pid + "__adapted_scheduled.csv")
            dff2.to_csv(self.analysisDir+"/" + self.pid + "_" + curTar +  "__adapted_scheduled.csv")


            #dff.column = dff.column.values/10000
            #dff2.column = dff2.column.values/10000

        
            fig, ax = plt.subplots(1,2, figsize=(10,5))
            #fig = plt.figure(figsize=(7,5))
            rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=99, l=55, sep=3, as_cmap=True)
            sns.heatmap(dff, cmap=rdgn, center=0.00, ax=ax[0])
            ax[0].set_title("TREATED-REF " + str(self.pid))
            sns.heatmap(dff2, cmap=rdgn, center=0.00, ax = ax[1])
            ax[1].set_title("ADAPTED-SCHEDULED")

            fig.suptitle('Target:' + curTar, fontsize=15)#, fontsize=30)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                    top=0.9, wspace=0.4,hspace=0.4)

            #FIXME
            fig.savefig(self.analysisDir + "/hm_pat_" + str(self.pid) + "_" + curTar + ".png", dpi=fig.dpi, bbox_inches='tight')
    
    
    def plotDVH(self, plotSess=None, vol=None, type=["ADAPTED_FROM", "REFERENCE_PLAN", "TREATED_PLAN"]):
        ### Referenzplan
        from matplotlib.lines import Line2D
        
        ii = self.ii
        sortedDataSess = self.sortedDataSess
        sortedData = self.sortedData
        
        #####
        if plotSess is None:
            plotSess = range(len(ii))
    
        NUM_COLORS = len(ii)
        #cm = plt.get_cmap('tab10')
        #cm = plt.get_cmap('Spectral')
        cm = ListedColormap(generate_colormap(NUM_COLORS*NUM_COLORS))
        fig = plt.figure()
        ax = plt.gca()
    
        names =  []
        for a in plotSess:
            i = ii[a][0][0]
            self.debug.info(sortedDataSess[i][0][0])
            print(".", end="")
            #print(sortedDataSess[i][0][0])
            
            if sortedDataSess[i][0][0] == sortedDataSess[i+1][0][0]:
                if sortedDataSess[i][0][0] == sortedDataSess[i+2][0][0]:
    
                    for z in range(3):#:type:
                        ## print?
                        cont = False
                        for zz in range(len(type)):
                            if type[zz] in sortedDataSess[i+z][1]:
                                cont = True
                                break
                        if not cont:
                            continue
    
                        tmp = sortedData[i+z]
                        if "ADAPTED_FROM" in sortedDataSess[i+z][1]:
                            linestyle = "-"
                        elif "REFERENCE_PLAN" in sortedDataSess[i+z][1]:
                            linestyle = "dashed"
                        elif "TREATED_PLAN" in sortedDataSess[i+z][1]:
                            linestyle = "dotted"
                        tmp = tmp.loc[:, ~tmp.columns.isin(['label', 'cc', 'mean'])]
                        tmp.index = sortedData[i+z]['label'].values
                        #divide x axis values 
                        #print([round(x,3) for x in tmp.columns.values/10000])
                        #tmp.columns = tmp.columns.values/10000
                        tmp.columns = [round(x,3) for x in tmp.columns.values/10000]
                        if not vol is None:
                            tmp=tmp.loc[tmp.index.values == vol,]
    
                        if len(plotSess) > 1 and not vol is None and len(vol) == 1:
                            tmp = tmp.set_index(pd.Index([ii[a][1]]))
                            names.append(ii[a][1])
                            tmp.T.plot(figsize=(12,8), linestyle=linestyle, ax=ax, color=cm(a/NUM_COLORS))
                        else:
                            tmp.T.plot(figsize=(12,8), linestyle=linestyle, ax=ax)
                    
                    if len(plotSess)  > 1 and not vol is None and len(vol) == 1:
                        #names = plotSess#sortedData[i]['label'].values.tolist()
                        print("",end="")
                    else:
                        names = sortedData[i]['label'].values.tolist()
    
                    for j, p in enumerate(ax.get_lines()):    # this is the loop to change Labels and colors
                        if p.get_label() in names[:j]:    # check for Name already exists
                            idx = names.index(p.get_label())       # find ist index
                            p.set_c(ax.get_lines()[idx].get_c())   # set color
                            p.set_label('_' + p.get_label())       # hide label in auto-legend
                    
                    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    
        
        data= [["-","ADAPTED_FROM"],["dashed","REFERENCE_PLAN"],["dotted", "TREATED_PLAN"]]
        lines = [Line2D([0], [0], color="black", linewidth=1, linestyle=x[0], label=x[1]) for x in data]
        legend2 = plt.legend(handles=lines,
                             bbox_to_anchor=(1.2, 0.2))
        plt.gca().add_artist(legend2)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        if not vol is None and len(vol) == 1:
            plt.title(self.pid + ": " + vol[0])
    
            
        ax.set_ylabel('Volume', fontsize=12)
        ax.set_xlabel('Dose [Gy]', fontsize=12)
    
        plt.savefig(self.analysisDir + "/pat_" + self.pid + "_plans.png", bbox_inches='tight')#, dpi=fig.dpi)
        plt.show()
        return(plt)


    ##### Create nifti from ETHOS data 
    def createNifti(self, session=None):
        sessPath = ""
        if not session is None:
            sessPath = session
    
        dcmpath = glob.glob(self.dcmpath + "/*/Session*/" + self.pid + "/"+ self.intent+ "/" + sessPath)[0] 
        
        rtstructs = []
        for root, dirs, files in os.walk(dcmpath, topdown=False):
            for name in files:
                if not "dcm" in name:
                    continue
                ### separate RT_STRUCT, CT, ...
                if name.startswith("RTSTRUCT."):
                    rtstructs.append([root, name])
    
        rtstructs_0 = []
        rtstructs_1 = []
        for i in range(len(rtstructs)):
            #print(rtstructs[i])
            rtstructs_0.append(rtstructs[i][0])
            rtstructs_1.append(rtstructs[i][1])
    
        
        baseout = self.basepath
    
        ###########################################
        #### extract niftis ... 
        extractedSessions = []
        for i in reversed(range(len(rtstructs_1))):
            print(".", end="")
            self.debug.info(str(i) + ": " + rtstructs_0[i])

            pid = rtstructs_0[i].split("/")[-3]
            organ = rtstructs_0[i].split("/")[-2].replace(" ", "_")
            session = rtstructs_0[i].split("/")[-1]
            session = session.replace(" ", "_")

            self.debug.info("PID: " + pid + " ORGAN " + organ + " SESSION: " + session)
            #print("PID: " + pid + " ORGAN " + organ + " SESSION: " + session)
    
            ### create outdir
            out = baseout + "/" + organ + "/" + pid + "/"+ session
            try:
                self.debug.info("##### CREATING NIFTI ... ####")
                ### create nifti
                niftiout = out + "/nifti/"
                os.system("mkdir -p " + niftiout)
                cmd = "dcm2niix -f '%p_%e_%s_%j'  -o '"+niftiout + "' '" + rtstructs_0[i]+"'"
                #print(cmd)
                self.debug.info(cmd)
                dir = os.listdir(niftiout)
                if len(dir) == 0:
                    os.system(cmd)
            except: 
                self.debug.info("Error")


    def calcMetrics(self, 
                    dose=[0.2, 0.5, 0.95, 0.99], 
                    funs=['max', 'mean', 'xx'], 
                    dose_vals=[5, 10, 20,30,40],
                    col=2):
        frames = self.frames
        metrics = self.metrics
        names=  self.names
        targets= self.targets
    
        coll = []
        ### doses  -> Dx values
        #### Min dose received by x% of the volume
        for do in dose:
            xx = [x.index.values[x['ADAPTED_FROM'] <= do] for x in frames]
            xx_adapted = [x[0] if len(x) > 0  else np.NAN for x in xx  ]
            xx = [x.index.values[x['TREATED_PLAN'] <= do] for x in frames]
            xx_treated = [x[0] if len(x) > 0  else np.NAN for x in xx  ]
            xx = [x.index.values[x['REFERENCE_PLAN'] <= do] for x in frames]
            xx_reference = [x[0] if len(x) > 0  else np.NAN for x in xx  ]
            dfX = pd.DataFrame({'adp': xx_adapted, 
                               'treated': xx_treated,
                               'ref': xx_reference})
            dfX = dfX/10000
            dfX['session'] = names
            dfX['sessID'] = [x.split("_")[1] for x in names]
            dfX['target'] = targets
            #frames[0].index.values[frames[0]['adapted'] > 0.2][0]
            sid = [x.split("_")[1] for x in names]
            dfXX = pd.DataFrame({'sessionID': np.concatenate([sid, sid, sid]),
                                'target': np.concatenate([targets, targets, targets]),
                                'val': np.concatenate([xx_adapted, xx_treated, xx_reference])/10000,
                                'type': np.concatenate([["adapted_from"]*len(xx_adapted), ["treated_plan"]*len(xx_treated), ["reference_plan"]*len(xx_reference )])
                                })
            dfXX['metr'] = "D"+ str(do*100)
            dfXX['unit'] = "Gy"
            coll.append(dfXX)

    
        ### Vx values
        ### Volume percentage that received at least x Gy
        for d in dose_vals:
            dbak = d
            #d=int(d*10000/int(rtm.fractions)) ##TODO!
            d=int(d*10000) ##TODO!
            # Für jeden Frame: den Volumenanteil bestimmen, der mindestens d Gy bekommt
            #vx_adapted = [x['ADAPTED_FROM'][x.index.values >= d].values[0] for x in frames]
            #vx_treated = [x['TREATED_PLAN'][x.index.values >= d].values[0] for x in frames]
            #vx_reference = [x['REFERENCE_PLAN'][x.index.values >= d].values[0] for x in frames]

            # Für jeden Frame: den Volumenanteil bestimmen, der mindestens d Gy bekommt
            vx_adapted = [
                x['ADAPTED_FROM'][x.index.values >= d].values[0] if np.any(x.index.values >= d) else 0
                for x in frames
            ]
            vx_treated = [
                x['TREATED_PLAN'][x.index.values >= d].values[0] if np.any(x.index.values >= d) else 0
                for x in frames
            ]
            vx_reference = [
                x['REFERENCE_PLAN'][x.index.values >= d].values[0] if np.any(x.index.values >= d) else 0
                for x in frames
            ]


            sid = [n.split("_")[1] for n in names]

            dfXX = pd.DataFrame({
            'sessionID': np.concatenate([sid, sid, sid]),
            'target': np.concatenate([targets, targets, targets]),
            'val': np.concatenate([vx_adapted, vx_treated, vx_reference]), 
            'type': np.concatenate([
                ["adapted_from"]*len(vx_adapted),
                ["treated_plan"]*len(vx_treated),
                ["reference_plan"]*len(vx_reference)
                ])
            })
            dfXX['metr'] = f"V{round(dbak,3)}"
            dfXX['unit'] = "%/100"
            coll.append(dfXX)




        ### metrics max, mean, vol
        dfM_CC = None
        for fun in funs:
            fact = 10000
            if fun == 'cc':
                fun = 'xx'
                fact = 10000
            metrMax_adapted = [x[fun][x['type'] == "ADAPTED_FROM"].values[0] for x in metrics]
            metrMax_treated = [x[fun][x['type'] == "TREATED_PLAN"].values[0] for x in metrics]
            metrMax_reference = [x[fun][x['type'] == "REFERENCE_PLAN"].values[0] for x in metrics]
            sid = [x.split("_")[1] for x in names]
            dfM_Max = pd.DataFrame({'sessionID': np.concatenate([sid, sid, sid]),
                                'target': np.concatenate([targets, targets, targets]),
                                'val': np.concatenate([metrMax_adapted, metrMax_treated, metrMax_reference])/fact,
                                'type': np.concatenate([["adapted_from"]*len(metrMax_adapted), ["treated_plan"]*len( metrMax_treated), ["reference_plan"]*len(metrMax_reference )])
                                })
            unit = "Gy"
            if fun == "xx":
                fun = "cc"
                unit = "cc"
            dfM_Max['metr'] = fun
            dfM_Max['unit'] = unit
            if fun == "cc": ### TODO: test 
                dfM_CC = dfM_Max

            coll.append(dfM_Max)

        ### D0.03cc (more robust than Dmax)
        vols = dfM_CC[dfM_CC['type'] == "adapted_from"]['val'] ### same for all palsn
        xx_adapted = [x[0].index.values[x[0]['ADAPTED_FROM']*x[1] <=0.03 ][0] for x in zip(frames, vols)]
        #xx_adapted = [x[0] if len(x) > 0  else np.NAN for x in xx  ]
        xx_treated = [x[0].index.values[x[0]['TREATED_PLAN']*x[1] <=0.03 ][0] for x in zip(frames, vols)]
        #xx_treated = [x[0] if len(x) > 0  else np.NAN for x in xx  ]
        xx_reference = [x[0].index.values[x[0]['REFERENCE_PLAN']*x[1] <=0.03 ][0] for x in zip(frames, vols)]
        #xx_reference = [x[0] if len(x) > 0  else np.NAN for x in xx  ]
        dfX = pd.DataFrame({'adp': xx_adapted, 
                               'treated': xx_treated,
                               'ref': xx_reference})
        dfX = dfX/10000
        dfX['session'] = names
        dfX['sessID'] = [x.split("_")[1] for x in names]
        dfX['target'] = targets
            #frames[0].index.values[frames[0]['adapted'] > 0.2][0]
        sid = [x.split("_")[1] for x in names]
        dfXX = pd.DataFrame({'sessionID': np.concatenate([sid, sid, sid]),
                                'target': np.concatenate([targets, targets, targets]),
                                'val': np.concatenate([xx_adapted, xx_treated, xx_reference])/10000,
                                'type': np.concatenate([["adapted_from"]*len(xx_adapted), ["treated_plan"]*len(xx_treated), ["reference_plan"]*len(xx_reference )])
                                })
        dfXX['metr'] = "D0.03cc"
        dfXX['unit'] = "Gy"
        coll.append(dfXX)
        ## 



        ######## save
    
        try:
            coll2 = [x for x in coll if x['metr'][0] != "cc"]
            pd.concat(coll).to_csv(self.analysisDir + "/dvhMetr_pat__"+ str(self.pid)+".csv")
            
            ###### extract trends
            mpg = dfM_CC[dfM_CC['type'] == "treated_plan"]
            mpg['sessionID'] = [int(x) for x in mpg['sessionID'].tolist()]
            mpg.to_csv(self.analysisDir + "/trends_pat__"+ str(self.pid)+".csv")
        except Exception as e:
            self.log.info("Error in calcMetrics()")
            self.log.info(e)
    




    
    def copyPath(self, fromPath, toPath, patName=None):
        self.log.info("Copy data from " + fromPath + " -> " + toPath)
        if not patName is None:
            fromPath = fromPath + "/" + patName + "/"
        else:
            self.debug.critical("No patname supplied!")
            raise
    
        if not os.path.isdir(fromPath):
            self.debug.critical("Path does not exist: " + fromPath)
            raise

        ### leading subfolder?
        fl = glob.glob(fromPath+"*")
        lead = ""
        foundSess = False
        for f in fl:
            if "Session" in f:
                foundSess = True
                lead= ""
                break
            else:
                lead = f.split("/")[-1]
    
        fromPath = fromPath + "/" + lead + "/"
        self.debug.info("Copy from " + fromPath)


        ### idenify session export path
        sessionPath  = glob.glob(fromPath+"Session*Export")[0]
        if os.path.isdir(sessionPath):
            self.debug.info("Session Path: " + sessionPath)
        else:
            self.debug.critial("Session path not found: " + sessionPath)
            raise
        sessionTo = toPath + "/" + patName + "/Session Export/"
            
        ## identify pid and organ
        fl = glob.glob(sessionPath + "/*/*/")[0]
        print(fl)
        pid = fl.split("/")[-3]
        org = fl.split("/")[-2]
        if not pid.isnumeric:
            tmp = pid 
            pid = organ
            organ = tmp
            if not pid.isnumeric:
                self.debug.critical("Organ: " + org + " PID: " + pid)
                raise
        self.debug.info("Organ: " + org + " PID: " + pid)


        ## copy sessions
        self.debug.info("Create " + toPath + "/" + patName)
        try:
            os.makedirs(toPath + "/" + patName, exist_ok=True)
            os.makedirs(sessionTo + "/"+pid + "/" + org + "/", exist_ok=True)
        except Exception as e:
            self.debug.info(e)
    
        for sess in glob.glob(sessionPath+"/"+pid + "/" + org + "/*"):
            self.debug.info("Copy " + sess + " to " + sessionTo + "/"+pid + "/" + org + "/" + sess.split("/")[-1])
            print(sess)
            if not os.path.isdir(sessionTo + "/"+pid + "/" + org + "/" + sess.split("/")[-1] + "/"):
                try:
                    os.makedirs( sessionTo + "/"+pid + "/" + org + "/" + sess.split("/")[-1],exist_ok=True)
                except Exception as e:
                    self.debug.info(e)
                try:
                    shutil.copytree(sess, sessionTo + "/"+pid + "/" + org + "/" + sess.split("/")[-1] + "/")
                except Exception as e:
                    self.debug.info(e)
            #if len(os.listdir(sessionTo + "/"+pid + "/" + org + "/" + sess.split("/")[-1] + "/")) == 0:
            src_files = os.listdir(sess)
            for file_name in src_files:
                full_file_name = os.path.join(sess, file_name)
                if os.path.isfile(full_file_name):
                    dest = sessionTo + "/"+pid + "/" + org + "/" + sess.split("/")[-1] + "/" + file_name
                    self.debug.info(full_file_name + " -> " + dest)
                    shutil.copyfile(full_file_name, dest)


    ######## compare DVHs
    def compare(self, targ=None):
        self.log.info("Compare DVHs ... ")
        ii = self.ii
        sortedDataSess = self.sortedDataSess
        sortedData = self.sortedData
        
        frames = []
        names= []
        targets = []
        metrics = []
        
        if targ is None:
            targ = sortedData[0]['label'].values

        for target in targ:
            self.log.debug(target)

            for j in range(len(ii)):
                try:
                    i = ii[j][0][0]
                    trg = self.match[target]
        
                    tmpp = []
                    
                    maxs = []
                    means = []
                    ccs = []
                    maxV = 0
                    for z in range(3):
                        tmp1 = sortedData[i+z]
                        tofind = tmp1['label'][tmp1['label'].isin(trg)].values[0]
                        mn1 = tmp1.loc[tmp1['label'] == tofind]['mean'].values[0]
                        cc1 = tmp1.loc[tmp1['label'] == tofind]['cc'].values[0]
                        tmp1 = tmp1.loc[tmp1['label'] == tofind].iloc[0,4:]
                        tmpp.append(tmp1)
                        maxs.append(max(tmp1.index.values))
                        means.append(mn1)
                        ccs.append(cc1)
                        maxV = max(np.append(np.array(tmp1.index.values), maxV))
                    
        
                    ## get range for max observed value and interpolate
                    rg = range(0,  round(maxV))
                    intt = []
                    typee = []
                    type0 = []
                    type1 = []
                    for z in range(3):
                        tmp1 = tmpp[z]
                        intt.append(np.interp(list(rg), tmp1.index.values.astype(float), tmp1.values.astype(float), right=0, left=1))
                        typee.append(sortedDataSess[i+z][1])
                        type0.append(sortedDataSess[i+z][1].split(":")[1])
                        type1.append(":".join(sortedDataSess[i+z][1].split(":")[0:2]))
        
                    diff = pd.DataFrame()
                    for z in range(3):
                        diff[typee[z]] = intt[z]
                        diff[type0[z]] = intt[z]
                        diff[type1[z]] = intt[z]
            
                    diff['TREATED-REF'] = diff['TREATED_PLAN']-diff['REFERENCE_PLAN']
                    diff['ADAPTED-SCHEDULED'] = diff['ADAPTED_FROM']-diff['REFERENCE_PLAN']
        
                    metr = pd.DataFrame({'type':type0,
                                         'type0':typee,
                                         'type1':type1,
                                         'max':maxs,
                                         'xx':ccs,
                                         'mean':means})
                    
                    metrics.append(metr)
                    frames.append(diff)
                    targets.append(target)
                    names.append(ii[j][1])
                except Exception as e:
                    #print(e)
                    self.debug.error(e)
        
        self.metrics = metrics
        self.frames = frames
        self.targets = targets
        self.names = names

    def exportInfo(self):
        """ Export information about patient:
            PID
            Total Dose + Number of fractions
            Names of plans 
            Number of adapted / scheduled plans
        """
        ## PID
        ## number of treated 
        #TODO

    def info(self):
        """ How to use
        
            Copy files
            rtm = RTMetrics(dcmpath, basepath, None)
            rtm.copyPath(fromPath, toPath, patName=patName)

            Extract masks & convert ot nifti 
            rtm = RTMetrics(dcmpath, basepath, pid)
            #rtm.copyPath(fromPath, toPath, patName=patName)
        """
    
    ######## preprcoessing
    def extractRTStruct(self, session=None):
        self.log.info("Extracting RTStruct ... ")

        sessPath = ""
        if not session is None:
            sessPath = session
    
        dcmpath = glob.glob(self.dcmpath + "/*/Session*/" + self.pid + "/"+self.intent+"/" + sessPath)[0]  
        rtstructs = []
        for root, dirs, files in os.walk(dcmpath, topdown=False):
            for name in files:
                if not "dcm" in name:
                    continue
                ### separate RT_STRUCT, CT, ...
                if name.startswith("RTSTRUCT."):
                    rtstructs.append([root, name])

        #print(rtstructs)
        #self.debug.info(rtstructs)
        
        rtstructs_0 = []
        rtstructs_1 = []
        for i in range(len(rtstructs)):
            rtstructs_0.append(rtstructs[i][0])
            rtstructs_1.append(rtstructs[i][1])
        
        baseout = self.basepath
        
        ####################################
        ### extract RTSTRUCTs 
        extractedSessions = []
        tmpSessions = {}
        for i in reversed(range(len(rtstructs_1))):
            self.debug.info(rtstructs_0[i])
            #print(rtstructs_0[i])
            self.debug.info(rtstructs_0[i])

            ## TODO: check ### FIXME
            pid = rtstructs_0[i].split("/")[-3]
            organ = rtstructs_0[i].split("/")[-2].replace(" ", "_")
            session = rtstructs_0[i].split("/")[-1]
            session = session.replace(" ", "_")
            
            self.debug.info("PID: " + pid + " ORGAN " + organ + " SESSION: " + session)
    
            ### create outdir
            out = baseout + "/" + organ + "/" + pid + "/"+ session
            try:
                self.debug.info ("######## EXTRACTING RTSTRUCT ... ####")
                #print ("######## EXTRACTING RTSTRUCT ... ####")
                ### extract RTSTRUCT
                rtout = out + "/RTSTRUCT/"
                os.system("mkdir -p " + rtout)
                self.debug.info(rtout)
                dir = os.listdir(rtout)
                self.debug.info(dir)
                ## check if already extracted
                nFiles = 0
                for ff in glob.glob(rtout + "/mask_*.nii.gz"):
                    nFiles = nFiles+1
                if nFiles == 0:
                    print("No Masks detected - continue")
                else:
                    continue
                
                if ".800." in rtstructs_1[i] and not session in extractedSessions: 
                    dcm = dcmread(rtstructs_0[i]+"/"+rtstructs_1[i])
                    print(dcm[0x3006,0x0020])
                    ### find sct for seriesid 
                    nmSCT = glob.glob(rtout + "/../nifti/sct*.nii")[0]
                    sct_series_id = "1.2.246"+ nmSCT.split("1.2.246")[-1].split(".nii")[0]
                    if "_" in sct_series_id:
                        sct_series_id = sct_series_id.split("_")[0]

                    #### remove enumerating characters
                    sct_series_id = re.sub("[^0-9.]", "", sct_series_id)
                    
                    #sct_series_id = nmSCT.split("_")[-1].split(".nii")[0]
                    self.debug.info("SCT ID: " + sct_series_id)
                    print("SCT Series ID: " + sct_series_id)
                    
                    l = []
                    if tmpSessions.get(session):
                        l = tmpSessions[session]
                    
                    if "SQ: Array" in str(dcm[0x3006,0x0020]):                    
                        dcmrtstruct2nii(rtstructs_0[i]+"/"+rtstructs_1[i], rtstructs_0[i]+ "/", rtout, series_id = sct_series_id)
                        extractedSessions.append(session)
                    else:
                        nSt = int(str(dcm[0x3006,0x0020]).split("length ")[-1].replace(">", "").replace("'", ""))
                        l.append([nSt, i, rtout, sct_series_id])
                    tmpSessions.update([(session, l)])
                        
    
            except Exception as e:
                self.debug.error(e)
    
        for sess in tmpSessions:
            if not sess in extractedSessions:
                print(sess)
                ### extraction not yet done
                tmp = tmpSessions[sess]
                ## identfy dcm with largest number of struct
                maxN = 0
                selI = 0
                for j in range(len(tmp)):
                    if tmp[j][0] > maxN:
                        maxN = tmp[j][0]
                        selI = j
                i = tmp[selI][1]
                rtout = tmp[selI][2]
                sct_series_id = tmp[selI][3]
                print("PARAM")
                print(rtstructs_0[i]+"/"+rtstructs_1[i])
                print(rtout)
                print(sct_series_id)
                dcmrtstruct2nii(rtstructs_0[i]+"/"+rtstructs_1[i], rtstructs_0[i]+"/", rtout, series_id = sct_series_id)
                extractedSessions.append(session)
        

    def checkFiles(self):
        """
        Check if all required RTDOSE and RTPLAN files are there.
        Check fi multiple treatmetn intents exists per PID
        """
        ## Treatment intents
        intents = {}
        for intent in glob.glob(self.dcmpath + "/*/Session Export/"+ self.pid + "/"+self.intent+"/"):
            curInt  = intent.split("/")[-2]
            self.log.info("PID: " + self.pid + " INTENT: " + curInt)
            if self.pid in intents.keys():
                if curInt != intents[self.pid]:
                    self.debug.critical("MULTIPLE INTENTS: " + self.pid + " " + curInt + " / " + intents[self.pid])
                    self.debug.critical("Please specify intent in RTMetrics constructor: RTMetrics(..., intent='')")
                    self.log.critical("MULTIPLE INTENTS: " + self.pid + " " + curInt + " / " + intents[self.pid])
                    self.log.critical("Please specify intent in RTMetrics constructor: RTMetrics(..., intent='')")
                    raise
            else:
                intents.update({self.pid:curInt})


        ## RTDOSE and RTPLAN
        for session in glob.glob(self.dcmpath + "/*/Session Export/"+ self.pid + "/"+self.intent+"/*"):
            print(".", end="")
            self.debug.info("SESSION: " + session)
            #sess = session.split("/")[8]
            sess = session.split("/")[-1]
            ### 3 RTPLANS
            count = 0
            for file in glob.glob(session + "/RTPLAN*"):
                size = os.path.getsize(file)
                if size == 0:
                    self.debug.critical("corrupt file, size 0: " + file)
                    raise
                count = count +1 
            if count != 3:
                self.debug.critical("Missing RTPLAN in " + session)
                raise
    
            ## 4 RTDOSES
            count = 0
            for file in glob.glob(session + "/RTDO*"):
                size = os.path.getsize(file)
                if size == 0:
                    self.debug.critical("corrupt file, size 0: " + file)
                    raise
                count = count +1 
            if count != 4:
                self.debug.critical("Missing RTDOSE in " + session)
                raise
            

    def countAdpSched(self):
        adapted = self.adapted
        scheduled = self.scheduled
        
        for s in self.tmsS:
            #print(s)
            v = self.tmsS[s]
            for vv in v:
                if v[vv][1] == 'TREATED_PLAN':
                    #print(v[vv])
                    if "/ADP" in v[vv][0]:
                        adapted = adapted+1
                    else:
                        scheduled = scheduled + 1
        
        self.adapted = adapted
        self.scheduled = scheduled
        print("ADAPTED: " + str(adapted))
        ret = {}
        ret.update({'adapted':adapted})
        ret.update({'scheduled':scheduled})

        ### TODO: check if adp+scjed == #fract
        return(ret)
    
    
    def identifyCBCT(self, allCBCT = False):
        tmsS = self.tmsS
        
        cbct = {}
        for session in tmsS:
            print(session)
            vals = tmsS[session]
            
            fls = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/nifti/*.nii")
            #print(fls)
            for i in range(len(fls)):
                #print(i)
                for v in vals:
                    if v in fls[i]:
                        #print(v)
                        fls[i] = ""
                if "Eq_1" in fls[i]:
                    fls[i] = ""
                if "sct_" in fls[i]:
                    fls[i] = ""
                if "1.2.246.352.800." in fls[i]:
                    fls[i] = ""
    
            fls = [x for x in fls if x != ""]
            #for i in range(len(fls)):
            #    print(fls[i])
    
            ## get only initial cb
            if not allCBCT:
                for i in range(len(fls)):
                    #print(fls[i])
                    if not "_2_1." in fls[i]:
                        fls[i] = ""
                fls = [x for x in fls if x != ""]
                #for i in range(len(fls)):
                #    print(fls[i])
                cbct.update({session:fls})
            else:
                cbct.update({session:fls})
    
        self.cbcts = cbct


    def identifySCT(self):                                                                                           
        tmsS = self.tmsS                                                                                             

        sct = {}
        for session in tmsS:
            print(session)
            vals = tmsS[session]

            fls = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/nifti/sct_*.nii")
            #print(fls)
            for i in range(len(fls)):
                if not "sct_" in fls[i]:
                    fls[i] = ""
                if "Eq_" in fls[i]:
                    fls[i] = ""
            fls = [x for x in fls if x != ""]
            sct.update({session:fls})

        self.scts = sct

    def identifyDose(self):
        tmsS = self.tmsS                                                                                                                                                                                                  
        doseAdp = {}
        doseSched = {}
        doseTreat = {}
        for session in tmsS:
            print(session)
            vals = tmsS[session]
            
            ## adapted
            for key in tmsS[session]:
                #print (key)
                #print(tmsS[session][key][1])
                
                fls = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/nifti/*" + key + "*.nii")
                if tmsS[session][key][1] == "TREATED_PLAN":
                    doseTreat.update({session:fls})
                elif tmsS[session][key][1] == "REFERENCE_PLAN":
                    doseSched.update({session:fls})
                elif tmsS[session][key][1] == "ADAPTED_FROM":
                    doseAdp.update({session:fls})
    
        self.doseAdp = doseAdp
        self.doseSched = doseSched
        self.doseTreat = doseTreat

    def calcRadiomics(self, masks=None, type='SCT'):
        sessions = glob.glob(self.basepath + "/*/" + self.pid + "/*")

        ## identify files FIXME
        #identifySCT()
        #identifyCBCT()
        #identifyDose()

        for session in sessions:
            self.log.info("Session: " + session)
            print(session)
            
            ###FIXME: loop
            if type == 'SCT':
                ctIn = self.sct[session.split("/")[-1]][0]
            elif type == 'CBCT':
                ctIn = self.cbcts[session.split("/")[-1]][0]
            elif type == 'Dtreat':
                ctIn = self.doseTreat[session.split("/")[-1]][0]
            elif type == 'Dsched':
                ctIn = self.doseSched[session.split("/")[-1]][0]
            elif type == 'Dadp':
                ctIn = self.doseAdp[session.split("/")[-1]][0]
            else:
                self.log.error("Unknown type for calcRadiomics! Values: SCT, CBCT, Dtreat, Dsched, Dadp. Provided value: " + str(type))
            
            if not os.path.isfile(ctIn):
                self.debug.error("Input CT does not exist: " + ctIn)
            for maskFile in glob.glob(session + "/RTSTRUCT/mask_*.nii.gz"):
                print(".", end="")
                nm = maskFile.split("/")[-1].split("mask_")[1].replace(".nii.gz","")
                if not masks is None:
                    if masks.count(nm) > 0:
                        self.debug.info(maskFile + " -> " + nm)
                        if type == 'SCT':
                            outFile = session + "/RTSTRUCT/feat/sct__" + nm + ".csv"
                        elif type == 'CBCT':
                            outFile = session + "/RTSTRUCT/feat/cbct__" + nm + ".csv"
                        elif type == 'Dtreat':
                            outFile = session + "/RTSTRUCT/feat/Dtreat__" + nm + ".csv"
                        elif type == 'Dsched':
                            outFile = session + "/RTSTRUCT/feat/Dsched__" + nm + ".csv"
                        elif type == 'Dadp':
                            outFile = session + "/RTSTRUCT/feat/Dadp__" + nm + ".csv"

                        self.debug.info("outFile: " + outFile)
                        if not os.path.isdir(session + "/RTSTRUCT/feat/"):
                            os.makedirs(session + "/RTSTRUCT/feat/")
                                             
                        if not os.path.isfile(outFile):
                            cmd = "pyradiomics --setting=label:255 --format csv --setting correctMask:True " + str(ctIn) + " " + str(maskFile) + " -o " + str(outFile)
                            self.debug.info(cmd)
                            os.system(cmd)
        

        

    #########
    ########
    #### registration 
    #########
    
    def command_iteration(self, method):
        print(
            f"{method.GetOptimizerIteration():3} "
            + f"= {method.GetMetricValue():10.5f} "
            + f": {method.GetOptimizerPosition()}"
        )
    
    def register(self, fixed, moving, tfmOut): 
        fixed = sitk.ReadImage(fixed, sitk.sitkFloat32)
        moving = sitk.ReadImage(moving, sitk.sitkFloat32)
        
        numberOfBins = 24
        samplingPercentage = 0.10
    
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfBins)
        R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, 200)
        R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
        R.SetInterpolator(sitk.sitkLinear)
        
        R.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(R))
    
        outTx = R.Execute(fixed, moving)
        
        print("-------")
        print(outTx)
        print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {R.GetOptimizerIteration()}")
        print(f" Metric value: {R.GetMetricValue()}")
        
        sitk.WriteTransform(outTx, tfmOut)
        
        #resampler = sitk.ResampleImageFilter()
        #resampler.SetReferenceImage(fixed)
        #resampler.SetInterpolator(sitk.sitkLinear)
        #resampler.SetDefaultPixelValue(-1000)
        #resampler.SetTransform(outTx)
        #out = resampler.Execute(moving)
        #sitk.WriteImage(out, "/tmp/moving_resampled.nii.gz")
    
    def registerSCT(self):
        sessions = glob.glob(self.basepath + "/*/" + self.pid + "/*")
    
        sno = [int(x.split("/")[-1].split("_")[1]) for x in sessions]
        sessions = [x for _, x in sorted(zip(sno, sessions))]
        
        ## reference session: 1
        ctFixed = self.scts['Session_1'][0]
        for session in sessions:
            self.log.info("Session: " + session)
            #ctMoving = session + "/RTSTRUCT/image.nii.gz" #### SCT
            ctMoving = self.scts[session.split("/")[-1]][0] #session + "/RTSTRUCT/image.nii.gz" #### SCT
            tfmOut = session + "/tfm/"
            pathlib.Path(tfmOut).mkdir(exist_ok=True)
            tfmOut = tfmOut + "sct_" + session.split("/")[-1] + "__to__sct_" + sessions[0].split("/")[-1] + ".tfm"
            if not pathlib.Path(tfmOut).is_file():
                print("fixed: " + ctFixed)
                print("moving: " + ctMoving)
                print("tfm: " + tfmOut)
                self.register(ctFixed, ctMoving, tfmOut)
    
    def registerCBCT(self):
        sessions = glob.glob(self.basepath + "/*/" + self.pid + "/*")
    
        sno = [int(x.split("/")[-1].split("_")[1]) for x in sessions]
        sessions = [x for _, x in sorted(zip(sno, sessions))]
        
        ## reference session: 1
        ctFixed = self.cbcts['Session_1'][0]
        print("ctFixed: " + str(ctFixed))
        for session in sessions:
            self.log.info("Session: " + session)
            #ctMoving = session + "/RTSTRUCT/image.nii.gz" #### SCT
            ctMoving = self.cbcts[session.split("/")[-1]][0] #session + "/RTSTRUCT/image.nii.gz" #### SCT
            tfmOut = session + "/tfm/"
            pathlib.Path(tfmOut).mkdir(exist_ok=True)
            tfmOut = tfmOut + "cbct_" + session.split("/")[-1] + "__to__cbct_" + sessions[0].split("/")[-1] + ".tfm"
            if not pathlib.Path(tfmOut).is_file():
                print("fixed: " + ctFixed)
                print("moving: " + ctMoving)
                print("tfm: " + tfmOut)
                self.register(ctFixed, ctMoving, tfmOut)
    
    
    def applyTransform(self, type="Dsched", ref="CBCT", defPixelVal=-1000): ### FIXME -> include ref for selection of transformations
        ## aggregate all doses for scheduled plans over sessions 
        tmsS = self.tmsS
    
        regFiles = {}
        for session in tmsS:
            print(session)
            vals = tmsS[session]
            #print(vals)
            movingFile = ""
            tfmFile = ""
            factor = ""
            
            if type.startswith("D"):
                for key in vals:
                    if type == "Dsched":
                        if vals[key][1] == "REFERENCE_PLAN":
                            print(key + " <- " + vals[key][1] + " factor: " + vals[key][2])
                            movingFile = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/nifti/*" + key +".nii")[0]
                            tfmFile = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/tfm/*")
                            tfmFile = [x for x in tfmFile if ref in x.upper()][0] 
                            factor =  vals[key][2]
                            break     
                    elif type == "Dtreat":
                        if vals[key][1] == "TREATED_PLAN":
                            print(key + " <- " + vals[key][1] + " factor: " + vals[key][2])
                            movingFile = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/nifti/*" + key +".nii")[0]
                            tfmFile = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/tfm/*")
                            tfmFile = [x for x in tfmFile if ref in x.upper()][0] 
                            factor =  vals[key][2]
                            break     
                    elif type == "Dadp":
                        if vals[key][1] == "ADAPTED_FROM":
                            print(key + " <- " + vals[key][1] + " factor: " + vals[key][2])
                            movingFile = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/nifti/*" + key +".nii")[0]
                            tfmFile = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/tfm/*")
                            tfmFile = [x for x in tfmFile if ref in x.upper()][0] 
                            factor =  vals[key][2]
                            break                         
            elif type == "SCT":
                movingFile = self.scts[session][0]
                #movingFile = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/RTSTRUCT/image.nii.gz")[0]
                tfmFile = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/tfm/*")
                tfmFile = [x for x in tfmFile if ref in x.upper()][0] 
            elif type == "CBCT": #FIXME
                movingFile = self.cbcts[session][0]
                tfmFile = glob.glob(self.basepath + "/*/" + self.pid + "/" + session + "/tfm/*")
                tfmFile = [x for x in tfmFile if ref in x.upper()][0] 

            print(movingFile)
            print(tfmFile)
            print(factor)
    
            moving = sitk.ReadImage(movingFile, sitk.sitkFloat32)
            tfm = sitk.ReadTransform(tfmFile)
            if type.startswith("dose_"):
                moving = moving*factor
    
            regFiles.update({session:[moving,tfm]})
    
    
        print("#############")
        fixed = regFiles['Session_1'][0]
        #sitk.WriteImage(fixed, "/tmp/fixed.nii.gz") ## FIXME
    
        outFiles = {}
        for file in regFiles:
            print(file)
            ## do the registration
            outTx = regFiles[file][1]
            print(outTx)
            moving = regFiles[file][0]
            #sitk.WriteImage(moving, "/tmp/moving.nii.gz")
    
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(defPixelVal) ## images: -1000; dose: 0 ### FIXME!!!
            resampler.SetTransform(outTx)
            out = resampler.Execute(moving)
            #sitk.WriteImage(out, "/tmp/out.nii.gz")
            outFiles.update({file:out})
    
        return(outFiles)

