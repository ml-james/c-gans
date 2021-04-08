# cWGAN-GP for the extended preliminary study on the CIFAR10 datset

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
import cv2
from functools import partial
from keras.models import Model
# deprecated in keras 2 higher
#from keras.layers.merge import _Merge
from keras.layers import Input,Conv2DTranspose,BatchNormalization,LeakyReLU,Dense,UpSampling2D,Conv2D,concatenate,Flatten,Reshape
from keras.preprocessing import image as image_
from keras.utils.np_utils import to_categorical

# Global Variables

path="..."
path_=".../results"
epochs = 250
condition_dim = 10
latent_dim = 144
training_ratio = 5
gradient_penalty_weight = 10
batch_size = 128

# Generator

class Generator(object):
    
    def __init__(self,latent_dim,condition_dim):
        
        generator_input_1 = Input(shape=(latent_dim,),name='g_1')
        generator_input_2 = Input(shape=(condition_dim,),name='g_2')
    
        generator_input = concatenate([generator_input_1,generator_input_2])

        x = Dense(1024)(generator_input)
        x = LeakyReLU()(x)
        x = Dense(128*8*8)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape((8,8,128,))(x)
        x = Conv2DTranspose(128,(5,5),strides=2,padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(64,(5,5),padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(64,(5,5),strides=2,padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(3,(5,5),padding='same',activation='tanh')(x)
        self.generator = Model(inputs=[generator_input_1,generator_input_2],outputs=[x,generator_input_2])
        
        #print(Generator_Model.summary())
        
    def get_model(self):
        return self.generator
    
# Discriminator Model

class Discriminator(object):
    
    def __init__(self,condition_dim):
        
        discriminator_input_1 = Input(shape=(32,32,3,),name='d_1')
        discriminator_input_2 = Input(shape=(condition_dim,),name='d_2')
        
        _discriminator_input_2 = Reshape((1,1,condition_dim))(discriminator_input_2)
        _discriminator_input_2 = UpSampling2D((32,32))(_discriminator_input_2)
        
        discriminator_input = concatenate([discriminator_input_1, _discriminator_input_2],name='is_it_here')
        
        x = Conv2D(64,(5,5),padding='same')(discriminator_input)
        x = LeakyReLU()(x)
        x = Conv2D(128,(5,5),kernel_initializer='he_normal',strides=[2,2],padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(128,(5,5),kernel_initializer='he_normal',strides=[2,2],padding='same')(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(1024,kernel_initializer='he_normal')(x)   
        x = LeakyReLU()(x)
        x = Dense(1,kernel_initializer='he_normal')(x)
        self.discriminator = Model(inputs=[discriminator_input_1,discriminator_input_2], outputs=x)
        
        #print(Discriminator_Model.summary())
        
    def get_model(self):
        return self.discriminator
    
# Wasserstein Defintions
        
class random_weighted_average(_Merge):
    def _merge_function(self,inputs):
        weights = K.random_uniform((batch_size,1,1,1))
        return (weights*inputs[0]) + ((1-weights) * inputs[1])
    
class random_weighted_average_(_Merge):
    def _merge_function(self,inputs):
        weights = K.random_uniform((batch_size,1))
        return (weights*inputs[0]) + ((1-weights)*inputs[1])
        
def wasserstein_loss(y_true,y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true,y_pred,averaged_samples,gradient_penalty_weight):
    gradients = K.gradients(K.sum(y_pred),averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1-gradient_l2_norm)
    return gradient_penalty

# ConditionalGAN
    
class ConditionalGAN(object):
    
    def __init__(self,latent_dim,condition_dim):
                
       # Generator
        
       g = Generator(latent_dim,condition_dim)
       self._generator = g.get_model()
        
       d = Discriminator(condition_dim)
       self._discriminator = d.get_model()
         
       for layer in self._discriminator.layers:
           layer.trainable = False
       self._discriminator.trainable = False

       cgan_input_1 = Input(shape=(latent_dim,))
       cgan_input_2 = Input(shape=(condition_dim,))
       cgan_output = self._discriminator(self._generator([cgan_input_1,cgan_input_2]))
       self._cgan = Model([cgan_input_1,cgan_input_2],cgan_output)
        
       # Compile Generator
        
       cgan_optimizer = keras.optimizers.Adam(lr=0.0001,beta_1=0.5,beta_2=0.9)
       self._cgan.compile(optimizer=cgan_optimizer,loss=wasserstein_loss)
        
       # Discriminator
                
       for layer in self._discriminator.layers:
           layer.trainable = True
       for layer in self._generator.layers:
           layer.trainable = False
       self._discriminator.trainable = True
       self._generator.trainable = False
        
       dis_input_1 = Input(shape=(32,32,3,))
       dis_input_2 = Input(shape=(10,))
       dis_input_3 = Input(shape=(latent_dim,))
        
       fake_image = self._generator([dis_input_3,dis_input_2])
       fake_decision = self._discriminator(fake_image)
       real_decision = self._discriminator([dis_input_1,dis_input_2])
       av_images = random_weighted_average()([dis_input_1,fake_image[0]])
       av_labels = random_weighted_average_()([dis_input_2,fake_image[1]])
       av_decision = self._discriminator([av_images,av_labels])
       
       partial_gp_loss = partial(gradient_penalty_loss,averaged_samples=av_images,gradient_penalty_weight=gradient_penalty_weight)
       partial_gp_loss.__name__ = 'Gradient_Penalty'
       self._discriminator_ = Model(inputs=[dis_input_1,dis_input_2,dis_input_3],outputs=[real_decision,fake_decision,av_decision])
       
       # Compile discriminator
       
       discriminator_optimizer = keras.optimizers.Adam(lr=0.0001,beta_1=0.5,beta_2=0.9)
       self._discriminator_.compile(optimizer=discriminator_optimizer,loss=[wasserstein_loss,wasserstein_loss,partial_gp_loss])
       
       print(self._discriminator_.summary())
       
    def train_gen(self,noise,condition,batch_size):
        
       misleading_targets = np.ones((batch_size,1))        
       g_loss = self._cgan.train_on_batch([noise,condition],misleading_targets)
        
       return g_loss
        
    def train_dis(self,image,condition,noise,batch_size):
                
       y_true = np.ones((batch_size,1),dtype=np.float32)
       y_false = -y_true
       y_dummy = np.zeros((batch_size,1),dtype=np.float32)
        
       d_loss = self._discriminator_.train_on_batch([image,condition,noise],[y_true,y_false,y_dummy])[0]
       
       return d_loss
        
    def predict(self,latent_vector,condition):
       return self._generator.predict([latent_vector,condition])[0]
        
    def load_weights(self,file_path,by_name=False):
        self._cgan.load_weights(file_path,by_name)
            
    def save_weights(self,file_path,overwrite=True):
        self._cgan.save_weights(file_path,overwrite)
        
    def save_model(self,file_path,overwrite=True):
        self._cgan.save(file_path,overwrite=True)
        
    def to_json(self,file_path,overwrite=True):
        self._cgan.to_json(file_path,overwrite)

# Main Program
        
#Training Data
        
def normalize(X):
    return (X-127.5)/127.5

def denormalize(X):
    return (X + 1.0)*127.5
    
def input_noise_data():
    
    (Z_Train,_),(_,_) = keras.datasets.cifar10.load_data()
    Z_Train_ = np.zeros([Z_Train.shape[0],144])
    for i in range(Z_Train.shape[0]):
        
        Z = cv2.resize((Z_Train[i]),(12,12))
        Z = np.ndarray.flatten(Z)
        
        B = []
        C = []
        D = []
        
        for j in range(Z.shape[0]):
            if j%3==0:
                B = np.append(B,Z[j])
            if j%3==1:
                C = np.append(C,Z[j])
            if j%3==2:
                D = np.append(D,Z[j])
        
        E = (B+C+D)/3
        E = (E - 127.5)/127.5
        
        Z_Train_[i] = E
            
    return Z_Train_

def input_noise_random():
    Z_Train = np.random.normal(0,1,(50000,144))
    return Z_Train

def make_training_data(condition_dim):

    (X_Train,Y_Train),(_,_) = keras.datasets.cifar10.load_data()
        
    X_Train = normalize(X_Train)
    Y_Train = to_categorical(Y_Train,condition_dim)

    Z_Train_Data = input_noise_data()
    Z_Train_Random = input_noise_random()
    
    return X_Train,Y_Train,Z_Train_Data,Z_Train_Random

# Train function

def train(latent_dim,condition_dim,epochs,path):
            
    # Get training data
        
    X_Train,Y_Train,Z_Train_Data,_ = make_training_data(condition_dim)
    
    X_Train = X_Train[0:17500]
    Y_Train = Y_Train[0:17500]
    Z_Train_Data = Z_Train_Data[0:17500]
    
    # Load CGAN
    
    cgan = ConditionalGAN(latent_dim,condition_dim)
        
    # Define empty arrays to store results
    
    d_loss_ = []
    discriminator_loss_ = []
    generator_loss_ = []
            
    for epoch in range(epochs):
        
                
        print('\n' + "Epoch: ", str(epoch+1))
    
        minibatches_size = batch_size * training_ratio
        
        for i in range(int(X_Train.shape[0] // (batch_size*training_ratio))):
            
            image_mb = X_Train[i*minibatches_size:(i+1)*minibatches_size]
            condition_mb = Y_Train[i*minibatches_size:(i+1)*minibatches_size]
            noise_mb = Z_Train_Data[i*minibatches_size:(i+1)*minibatches_size]
            
            for j in range(training_ratio):
                
                image = image_mb[j*batch_size:(j+1)*batch_size]
                condition = condition_mb[j*batch_size:(j+1)*batch_size]
                noise = noise_mb[j*batch_size:(j+1)*batch_size]
                
                d_loss = cgan.train_dis(image,condition,noise,batch_size)
                d_loss_ = np.append(d_loss_,d_loss)
            
            discriminator_loss = np.average(d_loss_)
    
            index = np.random.choice(len(Z_Train_Data),batch_size,replace=False)
            noise_ = Z_Train_Data[index]
            condition_ = Y_Train[index]
    
            generator_loss = cgan.train_gen(noise_,condition_,batch_size)
            
            if (i+1)%5 == 0:
                
                print("Training Loss:")
                    
                print("Discriminator Loss: ",format(discriminator_loss,".3f"))
                print("Generator Loss: ",format(generator_loss,".3f"))
                
                discriminator_loss_ = np.append(discriminator_loss_,discriminator_loss)
                generator_loss_ = np.append(generator_loss_,generator_loss)
                plt.figure(figsize=(10,8))
                plt.title("Loss vs Iterations")
                plt.xlabel('Units of 5 of Batch Size 128')
                plt.ylabel('Loss')
                plt.plot(discriminator_loss_,label='Discriminator Loss')
                plt.plot(generator_loss_,label='Generator Loss')
                plt.legend()
                plt.savefig(os.path.join(path,"Final_Loss_Graph.png"))
                plt.close()
                 
        if epoch%1 == 0:
            
            cgan.save_weights(os.path.join(path,'weights.h5'))
            rlv = np.random.randint(0,X_Train.shape[0])
            random_latent_vector = Z_Train_Data[rlv]
            random_latent_vector = np.reshape(random_latent_vector,(1,-1))
            condition = Y_Train[rlv]
            condition = np.reshape(condition,(1,-1))
            c = condition.argmax() 
            generated_image = cgan.predict(random_latent_vector,condition)[0]
            img = denormalize(generated_image)
            img = image_.array_to_img(img,scale=False)
            img.save(os.path.join(path,'g_d'+ str(epoch+1) + '_' + str(c+1) + '.png'))
        
        print('End of Epoch ' + str(epoch+1) + ':')
        print()
        
def predict(latent_dim,condition_dim,path_):
    
    cgan = ConditionalGAN(latent_dim,condition_dim)
    model_json = cgan._cgan.to_json()
    with open("generator_cifar.json", "w") as json_file:
        json_file.write(model_json)
    
    json_file = open('generator_cifar.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(path,"weights.h5"))
    model_ = Model(inputs=loaded_model.input,outputs=loaded_model.layers[2].get_output_at(1))
    
    _,input_label,input_noise,_ = make_training_data(condition_dim)
    n = 500
    
    for i in range(n):
        rlv = np.random.randint(0,input_noise.shape[0])
        random_latent_vector = input_noise[rlv]
        random_latent_vector = np.reshape(random_latent_vector,(1,-1))
        condition = input_label[rlv]
        condition = np.reshape(condition,(1,-1))
        c = condition.argmax() 
        generated_image = model_.predict([random_latent_vector,condition])[0]
        img = image_.array_to_img(denormalize(generated_image[0]),scale=False)
        img.save(os.path.join(path_,'g_after' + '_' + 'con' + str(c+1) + '_sample_' + str(i) + '.png'))      
            
if __name__ == "__main__":
    train(latent_dim,condition_dim,epochs,path) 
    predict(latent_dim,condition_dim,path_)
       
        





