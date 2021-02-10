#https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25 install gpu

#INIZIO codice per allenare la rete sulla cpu
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#FINE codice per allenare la rete sulla cpu

from tensorflow import keras
from keras_buoy.models import ResumableModel
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import  Dense, Conv3D, Dropout, Flatten, BatchNormalization 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from weights_extractor import SaveCompressedWeightsNetwork
from custom_model import createModel
from random import shuffle
import pickle
import math
import shutil
import argparse
#to plot the model
#from keras.utils.vis_utils import plot_model
#from keras.models import load_model
# Returns a compiled model identical to the saved one
#model = load_model('my_model.h5')

parser = argparse.ArgumentParser()
parser.add_argument('--resume',action='store_true',default=False)
parser.add_argument('--save-weights',action='store_true',default=False)
args = parser.parse_args()

PathSpectogramFolder=''
OutputPath=''
WeightsOutputPath=''
OutputPathModels=''
interictalSpectograms=[]
preictalSpectograms=[]  #This array contains syntetic data, it's created to have a balance dataset and it's used for training
preictalRealSpectograms=[]  #This array containt the real preictal data, it's used for testing
patients = ["01", "02", "05", "19", "21", "23"]
nSeizure=0

def loadParametersFromFile(filePath):
    global PathSpectogramFolder
    global OutputPath
    global WeightsOutputPath
    global OutputPathModels
    if(os.path.isfile(filePath)):
        with open(filePath, "r") as f:
                line=f.readline()
                if(line.split(":")[0]=="PathSpectogramFolder"):
                    PathSpectogramFolder=line.split(":")[1].strip()
                line=f.readline()
                if(line.split(":")[0]=="OutputPath"):
                    OutputPath=line.split(":")[1].strip()
                line=f.readline()
                if(line.split(":")[0]=="OutputPathModels"):
                    OutputPathModels=line.split(":")[1].strip()

def loadSpectogramData(indexPat):
    global interictalSpectograms
    global preictalSpectograms
    global preictalRealSpectograms
    global nSeizure
    nFileForSeizure=0
    
    interictalSpectograms=[]
    preictalSpectograms=[]
    preictalRealSpectograms=[]
    
    f = open(PathSpectogramFolder+'/paz'+patients[indexPat]+'/legendAllData.txt', 'r')
    line=f.readline()
    while(not "SEIZURE" in line):
        line=f.readline()
    nSeizure=int(line.split(":")[1].strip())
    line=f.readline()
    line=f.readline()#legge il numero di spectogrammi. non lo salvo dato che non mi serve
    nSpectograms=int(line.strip())
    nFileForSeizure=math.ceil(math.ceil(nSpectograms/50)/nSeizure)
    line=f.readline()#leggo il percorso del primo file
    
    #Lettura path files Interictal
    cont=-1
    indFilePathRead=0
    while("npy" in line and indFilePathRead<nSeizure*nFileForSeizure):
        if(indFilePathRead%nFileForSeizure==0):
            interictalSpectograms.append([])
            cont=cont+1
            interictalSpectograms[cont].append(line.split(' ')[2].rstrip())#.rstrip() remove \n
            indFilePathRead=indFilePathRead+1
        else:
            if(len(line.split(' '))>=3):
                interictalSpectograms[cont].append(line.split(' ')[2].rstrip())
            indFilePathRead=indFilePathRead+1
            
        line=f.readline()
    line=f.readline()#leggo PREICTAL
    line=f.readline()#leggo n° spectogram
    line=f.readline()#leggo n°seizure(SEIZURE X)

    #Lettura path files Preictal
    cont=-1
    indFilePathRead=0   
    #while(line and indFilePathRead<nSeizure*nFileForSeizure):    
    while(line.strip()!=""):
        if("SEIZURE" in line):
            line=f.readline()#ho letto n°seizure(SEIZURE X) perciò scorro in avanti
            if(len(line.split(' '))>=3):
                preictalSpectograms.append([])
                cont=cont+1
                preictalSpectograms[cont].append(line.split(' ')[2].rstrip())
                indFilePathRead=indFilePathRead+1
        else:
            if(len(line.split(' '))>=3):
                preictalSpectograms[cont].append(line.split(' ')[2].rstrip())
            indFilePathRead=indFilePathRead+1
            
        line=f.readline()
        
    line=f.readline()#leggo REAL_PREICTAL
    line=f.readline()#leggo n° spectogram
    line=f.readline()#leggo n°seizure(SEIZURE X)

    #Lettura path files Real Preictal
    cont=-1
    while(line):
        if("SEIZURE" in line):
            line=f.readline()#ho letto n°seizure(SEIZURE X) perciò scorro in avanti
            preictalRealSpectograms.append([])
            cont=cont+1
            preictalRealSpectograms[cont].append(line.split(' ')[2].rstrip())
        else:
            preictalRealSpectograms[cont].append(line.split(' ')[2].rstrip())
            
        line=f.readline()
    f.close()

# createModel() imported from custom_model.py

def getFilesPathWithoutSeizure(indexSeizure, indexPat):
    filesPath=[]
    for i in range(0, nSeizure):
        if(i!=indexSeizure):
            filesPath.extend(interictalSpectograms[i])
            filesPath.extend(preictalSpectograms[i])
    shuffle(filesPath)
    return filesPath


def generate_arrays_for_training(indexPat, paths, start=0, end=100):
    while True:
        from_=int(len(paths)/100*start)
        to_=int(len(paths)/100*end)
        for i in range(from_, int(to_)):
            f=paths[i]
            x = np.load(PathSpectogramFolder+f)
            #x=np.array([x])
            #x=x.swapaxes(0,1)
            x=np.expand_dims(x,-1)
            if('P' in f):
                y = np.repeat([[0,1]],x.shape[0], axis=0)
            else:
                y =np.repeat([[1,0]],x.shape[0], axis=0)
            yield((x*255).astype('uint8'),y)
            
def generate_arrays_for_predict(indexPat, paths, start=0, end=100):
    while True:
        from_=int(len(paths)/100*start)
        to_=int(len(paths)/100*end)
        for i in range(from_, int(to_)):
            f=paths[i]
            x = np.load(PathSpectogramFolder+f)
            #x=np.array([x])
            #x=x.swapaxes(0,1)
            x=np.expand_dims(x,-1)
            yield((x*255).astype('uint8'))

class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0, lower=True):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.lower=lower

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if self.lower:
            if current < self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True
        else:
            if current > self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True

def main():
    print("START")
    if not os.path.exists(OutputPathModels):
        os.makedirs(OutputPathModels)
    loadParametersFromFile("PARAMETERS_CNN.txt")
    #callback=EarlyStopping(monitor='val_acc', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)

    # custom early stop is incompatible with keras-buoy, but built-in early stop does not work for this project, falling back to custom early stop with val_acc
    earlystop=EarlyStoppingByLossVal(monitor='val_accuracy', value=0.975, verbose=1, lower=False)
    
    #earlystop = EarlyStopping(monitor='val_acc',min_delta=0,baseline=0.975,restore_best_weights=False,patience=0,mode='max',verbose=1)
        
    # no need for modelcheckpoint callback
    # modelcheckpoint = ModelCheckpoint(f'{OutputPathModels}/checkpoints/checkpoint_.h5',monitor='val_accuracy',verbose=1,save_best_only=False,save_weights_only=False)
    print("Parameters loaded")
    
    if args.resume and os.path.exists(f'{OutputPathModels}/resume_indices.txt'):
        with open(f'{OutputPathModels}/resume_indices.txt','r') as resf:
            resind = resf.readline()
            indPat = int(resind.split('.')[0])
            indi = int(resind.split('.')[1])
    else:
        indPat = 0
        indi = 0

    for indexPat in range(indPat, len(patients)):
        print('Patient '+patients[indexPat])
        if not os.path.exists(OutputPathModels+"ModelPat"+patients[indexPat]+"/"):
            os.makedirs(OutputPathModels+"ModelPat"+patients[indexPat]+"/")
            os.makedirs(OutputPathModels+"ModelPat"+patients[indexPat]+"/checkpoints/")
        else:
            if not args.resume:
                print('You said to not resume, deleting checkpoints...')
                # This will delete the pkl, h5 and txt file used to store checkpoints
                if os.listdir(OutputPathModels+"ModelPat"+patients[indexPat]+"/checkpoints/"):
                    os.remove(os.path.join(OutputPathModels,'resume_indices.txt'))
                    for f in os.listdir(OutputPathModels+"ModelPat"+patients[indexPat]+"/checkpoints/"):
                        os.remove(os.path.join(OutputPathModels+"ModelPat"+patients[indexPat]+"/checkpoints/",f))
        
        loadSpectogramData(indexPat) 
        print('Spectograms data loaded')
        
        result='Patient '+patients[indexPat]+'\n'     
        result='Out Seizure, True Positive, False Positive, False negative, Second of Inter in Test, Sensitivity, FPR \n'
        for i in range(indi, nSeizure):
            print('SEIZURE OUT: '+str(i+1))
            with open(f'{OutputPathModels}/resume_indices.txt','w') as resf:
              resf.write(f'{indexPat}.{i}')
            # create the model and make it resumable
            model = createModel()
            resumable_model = ResumableModel(model,save_every_epochs=1,to_path=f'{OutputPathModels}ModelPat{patients[indexPat]}/checkpoints/model_checkpoint_s{i}.h5')
            
            finalWeightsOutputPath=WeightsOutputPath+f'/paz{indexPat+1}/seizure{i+1}'
            if args.save_weights:
                # Define output dir for weights based on patient and seizure
                weights_callback = SaveCompressedWeightsNetwork(finalWeightsOutputPath,resume=args.resume)
                print('You asked to save weights, adding callback...')
                callback = [earlystop,weights_callback]
            else:
                print('Defaulting callback to earlystop...')
                callback = [earlystop]
            print('Training start')  
            filesPath=getFilesPathWithoutSeizure(i, indexPat)
            
            history = resumable_model.fit(generate_arrays_for_training(indexPat, filesPath, end=75), #end=75),#It take the first 75%
                                validation_data=generate_arrays_for_training(indexPat, filesPath, start=75),#start=75), #It take the last 25%
                                #steps_per_epoch=10000, epochs=10)
                                steps_per_epoch=int((len(filesPath)-int(len(filesPath)/100*25))),#*25), 
                                validation_steps=int((len(filesPath)-int(len(filesPath)/100*75))),#*75),
                                verbose=1, #no progress bar -> faster
                                epochs=300, max_queue_size=2, shuffle=True, callbacks=callback)# 100 epochs è meglio #aggiungere criterio di stop in base accuratezza
            print('Training end')

            # Save model history
            with open(f'{finalWeightsOutputPath}_hist.pkl','wb') as h:
                pickle.dump(history,h)

            print('Testing start')
            filesPath=interictalSpectograms[i]
            interPrediction=model.predict_generator(generate_arrays_for_predict(indexPat, filesPath), max_queue_size=4, steps=len(filesPath))
            filesPath=preictalRealSpectograms[i]
            preictPrediction=model.predict_generator(generate_arrays_for_predict(indexPat, filesPath), max_queue_size=4, steps=len(filesPath))
            print('Testing end')
            

            # Creates a HDF5 file 
            model.save(OutputPathModels+"ModelPat"+patients[indexPat]+"/"+'ModelOutSeizure'+str(i+1)+'.h5')
            print("Model saved")
            
            #to plot the model
            #plot_model(model, to_file="CNNModel", show_shapes=True, show_layer_names=True)
            
            if not os.path.exists(OutputPathModels+"OutputTest"+"/"):
                os.makedirs(OutputPathModels+"OutputTest"+"/")
            np.savetxt(OutputPathModels+"OutputTest"+"/"+"Int_"+patients[indexPat]+"_"+str(i+1)+".csv", interPrediction, delimiter=",")
            np.savetxt(OutputPathModels+"OutputTest"+"/"+"Pre_"+patients[indexPat]+"_"+str(i+1)+".csv", preictPrediction, delimiter=",")
            
            secondsInterictalInTest=len(interictalSpectograms[i])*50*30#50 spectograms for file, 30 seconds for each spectogram
            acc=0#accumulator
            fp=0
            tp=0
            fn=0
            lastTenResult=list()
            
            for el in interPrediction:
                if(el[1]>0.5):
                    acc=acc+1
                    lastTenResult.append(1)
                else:
                    lastTenResult.append(0)
                if(len(lastTenResult)>10):
                    acc=acc-lastTenResult.pop(0)
                if(acc>=8):
                  fp=fp+1
                  lastTenResult=list()
                  acc=0
            
            lastTenResult=list()
            for el in preictPrediction:
                if(el[1]>0.5):
                    acc=acc+1
                    lastTenResult.append(1)
                else:
                    lastTenResult.append(0)
                if(len(lastTenResult)>10):
                    acc=acc-lastTenResult.pop(0)
                if(acc>=8):
                  tp=tp+1 
                else:
                    if(len(lastTenResult)==10):
                       fn=fn+1 
                       
            sensitivity=tp/(tp+fn)
            FPR=fp/(secondsInterictalInTest/(60*60))
            
            result=result+str(i+1)+','+str(tp)+','+str(fp)+','+str(fn)+','+str(secondsInterictalInTest)+','
            result=result+str(sensitivity)+','+str(FPR)+'\n'
            print('True Positive, False Positive, False negative, Second of Inter in Test, Sensitivity, FPR')
            print(str(tp)+','+str(fp)+','+str(fn)+','+str(secondsInterictalInTest)+','+str(sensitivity)+','+str(FPR))

        indi = 0
        with open(OutputPath, "a+") as myfile:
            myfile.write(result)
    

if __name__ == '__main__':
    main()
    
