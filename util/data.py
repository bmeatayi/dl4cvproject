import hdf5storage
import numpy as np
import cv2
import time
import matplotlib.pylab
from scipy import misc
import torch
import random
from fixation_extraction import FixationLoader 

%load_ext autoreload
%autoreload 2


class dataPreprocessing():
    
    def __init__(self):
        #self.path=path # path to dataset
        self.classes = 3
        self.FramesPerClip = 16
        self.stride = 1
        self.desired_width = 112
        self.desired_height = 112
        self.human = []
        self.animal = []
        self.manmade = []
        
        self.human_train = []
        self.human_test = []
        self.human_valid = []
        
        self.animal_train = []
        self.animal_test = []
        self.animal_valid = []
        
        self.manmade_train = []
        self.manmade_test = []
        self.manmade_valid = []
        
        self.all_train_videos = np.zeros((0,self.desired_height,self.desired_width,3))
        #self.suma = np.zeros((self.desired_height,self.desired_width,3))
        #self.sumh = np.zeros((self.desired_height,self.desired_width,3))
        #self.summ = np.zeros((self.desired_height,self.desired_width,3))
        #self.sum = np.zeros((self.desired_height,self.desired_width,3))
        #self.square_suma = np.zeros((self.desired_height,self.desired_width,3))
        #self.square_sumh = np.zeros((self.desired_height,self.desired_width,3))
        #self.square_summ = np.zeros((self.desired_height,self.desired_width,3))
        #self.square_sum = np.zeros((self.desired_height,self.desired_width,3))
        self.numa = 0
        self.numh = 0
        self.numm = 0
        self.num = 0
        
        self.mean = np.zeros((self.desired_height,self.desired_width,3))
        self.mean_of_squares = np.zeros((self.desired_height,self.desired_width,3))
        self.std = np.zeros((self.desired_height,self.desired_width,3))
        
        #read video names
        x = hdf5storage.loadmat('VideoNameList.mat')
       
        arrays={}
       
        for k, v in x.items():
            arrays[k] = np.array(v)
           
        self.VideoNames = arrays[k]
        
        #read video info
        #VideoInfoMat = hdf5storage.loadmat('VideoInfo.mat')
        #arraysI={}

        #for k, v in VideoInfoMat.items():
        #    arraysI[k] = np.array(v)
        #self.VideoInfo = arrays[k] 
        #self.Video_num_frames = self.VideoInfo[:,2]
    
    
    def get_num_Videos(self):
        return len(self.VideoNames)
    
    def get_Video_Names(self):
        return self.VideoNames
    
    def divide_classes(self):
        for i in range(len(self.VideoNames)):
            name = self.VideoNames[i][0][0][0]
            if "animal" in name:
                self.animal.append(name)
            elif "human" in name:
                self.human.append(name)
            else:
                self.manmade.append(name)
   
        animal_shuffled = self.animal[:]
        random.shuffle(animal_shuffled) 
        human_shuffled = self.human[:]
        random.shuffle(human_shuffled)
        manmade_shuffled = self.manmade[:]
        random.shuffle(manmade_shuffled)
        
        self.animal_train = animal_shuffled[:len(self.animal)*60//100]
        self.animal_test = animal_shuffled[len(self.animal)*60//100 : len(self.animal)*80//100]
        self.animal_valid = animal_shuffled[len(self.animal)*80//100:]
        
        self.human_train = human_shuffled[:len(self.human)*60//100]
        self.human_test = human_shuffled[len(self.human)*60//100 : len(self.human)*80//100]
        self.human_valid = human_shuffled[len(self.human)*80//100:]
        
        self.manmade_train = manmade_shuffled[:len(self.manmade)*60//100]
        self.manmade_test = manmade_shuffled[len(self.manmade)*60//100 : len(self.manmade)*80//100]
        self.manmade_valid = manmade_shuffled[len(self.manmade)*80//100:]
        
        print(len(self.animal_train),len(self.animal_test),len(self.animal_valid))
        print(len(self.human_train),len(self.human_test),len(self.human_valid))        
        print(len(self.manmade_train),len(self.manmade_test),len(self.manmade_valid))
   

    def read_train(self):
        
        #for human
        #for train
        tic=time.clock()
        self.human_vids_train = []
        for vid in self.human_train:
            VideoFullName = 'human/'+vid + '/'+vid+'.mp4'
            buf = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
            self.human_vids_train.append(buf)
            #self.all_train_videos = np.concatenate([self.all_train_videos,np.double(buf)])
            #self.sumh += np.sum(buf,0)
            #self.square_sumh += np.sum(np.square(buf),0)
            self.numh += buf.shape[0]
            print(vid,self.numh)
            
        toc=time.clock()
        proc_time=toc-tic
        print('Reading human trainSet :',proc_time)
        
        
        #for animals
        #for train
        tic=time.clock()
        self.animal_vids_train = []
        for vid in self.animal_train:
            VideoFullName = 'animal/'+vid + '/'+vid+'.mp4'
            buf = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
            self.animal_vids_train.append(buf)
            #self.all_train_videos = np.concatenate([self.all_train_videos,np.double(buf)])
            #self.suma += np.sum(buf,0)
            #self.square_suma += np.sum(np.square(buf),0)
            self.numa += buf.shape[0]
            print(vid,self.numa)
            
        
        toc=time.clock()
        proc_time=toc-tic
        print('Reading animal trainSet :',proc_time)
        

        
        #for manmade
        #For train
        tic=time.clock()
        self.manmade_vids_train = []
        for vid in self.manmade_train:
            VideoFullName = 'manmade/'+vid + '/'+vid+'.mp4'
            buf = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
            self.manmade_vids_train.append(buf)
            #self.all_train_videos = np.concatenate([self.all_train_videos,np.double(buf)])
            #self.summ += np.sum(buf,0)
            #self.square_summ += np.sum(np.square(buf),0)
            self.numm += buf.shape[0]
            print(vid,self.numm)
            
        toc=time.clock()
        proc_time=toc-tic
        print('Reading manmade trainSet :',proc_time)    

        
        
    def calculate_mean_and_std(self):
        self.num = self.numa + self.numh + self.numm
        #for animals
        tic=time.clock()
        for idx, vid in enumerate(self.animal_train):
            buf = self.animal_vids_train[idx]
            #buf = np.double(buf)
            self.mean += np.sum(np.double(buf),0)/self.num
            self.mean_of_squares += np.sum(np.square(np.double(buf)),0)/self.num
            print(vid)            
        
        toc=time.clock()
        proc_time=toc-tic
        print('mean animal trainSet :',proc_time)
        
        #for human
        tic=time.clock()
        for idx, vid in enumerate(self.human_train):
            buf = self.human_vids_train[idx]
            #buf = np.double(buf)
            self.mean += np.sum(np.double(buf),0)/self.num
            self.mean_of_squares += np.sum(np.square(np.double(buf)),0)/self.num
            print(vid)
            
        toc=time.clock()
        proc_time=toc-tic
        print('mean human trainSet :',proc_time)
        
        #for manmade
        tic=time.clock()
        for idx, vid in enumerate(self.manmade_train):
            buf = self.manmade_vids_train[idx]
            #buf = np.double(buf)
            self.mean += np.sum(np.double(buf),0)/self.num
            self.mean_of_squares += np.sum(np.square(np.double(buf)),0)/self.num
            print(vid)
            
        toc=time.clock()
        proc_time=toc-tic
        print('mean manmade trainSet :',proc_time)        
        
        #def normalize_and_save(self):        
        #calculate mean and std of training dataset
        #mean = np.mean(self.all_train_videos,axis=0)
        #std = np.std(self.all_train_videos,axis=0)
        #self.num = self.numa + self.numh + self.numm
        #mean = self.suma/self.sum + self.sumh/self.num + self.summ/self.num#self.sum/self.num
        #mean_of_square = self.square_suma/self.num + self.square_sumh/self.num + self.square_summ/self.num
        self.std = np.sqrt(self.mean_of_squares - self.mean)
        matplotlib.pylab.imshow(self.mean)
        matplotlib.pylab.show()
        matplotlib.pylab.imshow(self.mean_of_squares)
        matplotlib.pylab.show()
        matplotlib.pylab.imshow(self.std)
        matplotlib.pylab.show()
        
        
    def save_all(self):    
        ################### now lets save these videos
        #trinset
        tic=time.clock()
        temp_animal_train = self.animal_vids_train[:]
        for idx, vid in enumerate(self.animal_train):
            normalized = (np.double(temp_animal_train[idx]) - self.mean)/self.std
            np.save('TrainSet/videos/'+vid,normalized)
            xx = FixationLoader()
            fixation_data = xx.get_video_fixation('animal/'+vid+'/Data.mat')
            np.save('TrainSet/groundtruth/'+vid, fixation_data)
            
        temp_human_train = self.human_vids_train[:]
        for idx, vid in enumerate(self.human_train):
            normalized = (np.double(temp_human_train[idx]) - self.mean)/self.std
            np.save('TrainSet/videos/'+vid,normalized)
            xx = FixationLoader()
            fixation_data = xx.get_video_fixation('human/'+vid+'/Data.mat')
            np.save('TrainSet/groundtruth/'+vid, fixation_data)
            
        temp_manmade_train = self.manmade_vids_train[:]
        for idx, vid in enumerate(self.manmade_train):
            normalized = (np.double(temp_manmade_train[idx]) - self.mean)/self.std
            np.save('TrainSet/videos/'+vid,normalized)
            xx = FixationLoader()
            fixation_data = xx.get_video_fixation('manmade/'+vid+'/Data.mat')
            np.save('TrainSet/groundtruth/'+vid, fixation_data)
            
        toc=time.clock()
        proc_time=toc-tic
        print('Saving trainSet :',proc_time)
        
        #testset
        tic=time.clock()
        for vid in self.animal_test:
            VideoFullName = 'animal/'+vid + '/'+vid+'.mp4'
            buf = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
            normalized = (np.double(buf) - self.mean)/self.std
            np.save('TestSet/videos/'+vid,normalized)
            xx = FixationLoader()
            fixation_data = xx.get_video_fixation('animal/'+vid+'/Data.mat')
            np.save('TestSet/groundtruth/'+vid, fixation_data)
            
        for vid in self.human_test:
            VideoFullName = 'human/'+vid + '/'+vid+'.mp4'
            buf = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
            normalized = (np.double(buf) - self.mean)/self.std
            np.save('TestSet/videos/'+vid,normalized)
            xx = FixationLoader()
            fixation_data = xx.get_video_fixation('human/'+vid+'/Data.mat')
            np.save('TestSet/groundtruth/'+vid, fixation_data)

        for vid in self.manmade_test:
            VideoFullName = 'manmade/'+vid + '/'+vid+'.mp4'
            buf = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
            normalized = (np.double(buf) - self.mean)/self.std
            np.save('TestSet/videos/'+vid,normalized)
            xx = FixationLoader()
            fixation_data = xx.get_video_fixation('manmade/'+vid+'/Data.mat')
            np.save('TestSet/groundtruth/'+vid, fixation_data)    
        
        toc=time.clock()
        proc_time=toc-tic
        print('Reading & Saving testSet :',proc_time)
            
        #validset
        tic=time.clock()
        for vid in self.animal_valid:
            VideoFullName = 'animal/'+vid + '/'+vid+'.mp4'
            buf = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
            normalized = (np.double(buf) - self.mean)/self.std
            np.save('ValidSet/videos/'+vid,normalized)
            xx = FixationLoader()
            fixation_data = xx.get_video_fixation('animal/'+vid+'/Data.mat')
            np.save('ValidSet/groundtruth/'+vid, fixation_data)
            
        for vid in self.human_valid:
            VideoFullName = 'human/'+vid + '/'+vid+'.mp4'
            buf = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
            normalized = (np.double(buf) - self.mean)/self.std
            np.save('ValidSet/videos/'+vid,normalized)
            xx = FixationLoader()
            fixation_data = xx.get_video_fixation('human/'+vid+'/Data.mat')
            np.save('ValidSet/groundtruth/'+vid, fixation_data)

        for vid in self.manmade_valid:
            VideoFullName = 'manmade/'+vid + '/'+vid+'.mp4'
            buf = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
            normalized = (np.double(buf) - self.mean)/self.std
            np.save('ValidSet/videos/'+vid,normalized)
            xx = FixationLoader()
            fixation_data = xx.get_video_fixation('manmade/'+vid+'/Data.mat')
            np.save('ValidSet/groundtruth/'+vid, fixation_data)    
        
        toc=time.clock()
        proc_time=toc-tic
        print('Reading & Saving validSet :',proc_time)
            
        
        
        
        
    def read_video(self,VideoFullName, FramesPerClip, stride, desired_width, desired_height):
        cap = cv2.VideoCapture(VideoFullName)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #buf = torch.FloatTensor(frameCount, 3, desired_height, desired_width)
        buf = np.zeros((frameCount, desired_height, desired_width,3), np.dtype('uint8'))
        frame_temp = np.empty((frameHeight, frameWidth,3), np.dtype('uint8'))        
        
        fc = 0
        ret = True
        while (fc < frameCount  and ret):   
            ret, frame_temp = cap.read()
            frame=misc.imresize(frame_temp, [desired_height, desired_width], interp='cubic', mode=None)
            #frame = torch.from_numpy(frame)
            buf[fc]=frame
            fc += 1
        cap.release()
        #bufx = np.double(buf)
        return buf#x
    
'''
    
if __name__ == '__main__':
    
    data = dataPreprocessing()
    data.divide_classes()
    data.all_videos()
    
'''    
