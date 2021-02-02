import numpy as np
import tensorflow as tf
import pickle
import os
import shutil

# This class helps in saving the weights of a Dense/Convolutional layer for each batch iteration during training
# For Dense layer, saves only the weights and ignore the bias
class SaveCompressedWeightsNetwork(tf.keras.callbacks.Callback):
  def __init__(self, output_dir):
    super(tf.keras.callbacks.Callback).__init__()
    if not os.path.exists(output_dir): # Directory does not exists -> create the directory path
    	os.makedirs(output_dir)
    else:
    	if os.listdir(output_dir): # Directory exists and is not empty -> output warning and continue
    	    print('Output directory already exists and is not empty, existing files will be overwritten.')
    self.output_dir = output_dir
  def on_train_begin(self, batch, logs=None):
    self.epochs = self.params.get('epochs')
    self.initial = {n:1 for n in range(len(self.model.layers))}
    self.weights_layer = {n:[] for n in range(len(self.model.layers))}
    # Saves weight initialization
    for n in range(len(self.model.layers)):
      # Exclude layers which does not have weights (Dropout, Flatten, MaxPool...)
      if self.model.layers[n].name in ['conv','dense']:
        self.initial[n] *= self.model.layers[n].get_weights()[0]
    # pickle.dump(self.initial,open(f'initial.pkl','wb'))

  def on_train_batch_end(self, batch, logs=None):
    # Saves all layers' weights at the end of each batch
    for n in range(len(self.model.layers)):
      if self.model.layers[n].name in ['conv','dense']:
        self.weights_layer[n].append(self.model.layers[n].get_weights()[0])
  
  def on_epoch_end(self, epoch, logs=None):
    # Saves weights splitting per epochs. The first epoch will contain weights initialization at first position
    if epoch == 0: # Put weight initialization as first element in weights layer
      for n in range(len(self.model.layers)):
        self.weights_layer[n].insert(0,self.initial[n])
    save_fname = str.zfill(str(epoch+1),len(str(self.epochs))+1)
    
    pickle.dump(self.weights_layer,open(f'{os.path.join(self.output_dir,save_fname)}.pkl','wb'))
    # Reset arrays for layers' weight for next epoch
    for n in range(len(self.model.layers)):
      self.weights_layer[n] = []
