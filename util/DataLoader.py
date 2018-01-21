import hdf5storage
import numpy as np
import cv2
import time
import matplotlib.pylab
from scipy import misc
import torch.utils.data as data

import os
import torch

class vidsalDataset(data.Dataset):
    """Dataset class for videos
       Put video files in the folder /videos and fixation data  in the folder /groundtruth
       
    """
    def __init__(self, dir_path):
        self.dir_path_vid = dir_path + 'videos/'
        self.dir_path_gt = dir_path + 'groundtruth/'
        filelist = os.listdir(self.dir_path_vid)
        
        #Make sure that files other than videos are not included in the list
        for file in filelist:
            if file.startswith('.'):
                filelist.remove(file)
        self.filelist = filelist
        
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        vid_path = self.dir_path_vid + self.filelist[idx]
        gt_path = self.dir_path_gt + self.filelist[idx]
        vid = np.load(vid_path)
        groundtruth = np.load(gt_path)
        return vid,groundtruth
    

class dataset():
    
    def __init__(self):
        #self.path=path # path to dataset
        self.FramesPerClip = 16
        self.stride = 1
        self.desired_width = 112
        self.desired_height = 112

        self.ideoNameMat = hdf5storage.loadmat('VideoNameList.mat')
        
        arrays={}

        for k, v in VideoNameMat.items():
            arrays[k] = np.array(v)
        self.VideoNames = arrays[k]

    
    def get_num_Videos():
        return len(self.VideoNames)
    
    
    def get_Video_Names():
        return self.VideoNames
    
    
    def read_video(VideoFullName, FramesPerClip, stride, desired_width, desired_height):
        cap = cv2.VideoCapture(VideoFullName)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = torch.FloatTensor(frameCount, 3, desired_height, desired_width)
        frame_temp = np.empty((frameHeight, frameWidth, 3), np.dtype('uint8'))
        
        NumOfClips = (frameCount - FramesPerClip + stride)//stride
        frameShift = FramesPerClip - stride

        clips = torch.FloatTensor(NumOfClips,FramesPerClip, 3, desired_height, desired_width)

        fc = 0
        ret = True
        while (fc < frameCount  and ret):   
            ret, frame_temp = cap.read()
            frame=misc.imresize(frame_temp, [desired_height, desired_width], interp='cubic', mode=None)
            frame = torch.from_numpy(frame)
            buf[fc]=frame
            
            if (fc > frameShift-1):
                clips[fc-frameShift] = buf[fc-frameShift:fc+1]
            fc += 1
        cap.release()
        return buf, clips
    
    
    def get_video(video_id):
        tic = time.clock()
        VideoName = self.VideoNames[video_id][0][0][0]
        VideoFullName = VideoName+'/'+VideoName+'.mp4'
        video, clips = self.read_video(VideoFullName, self.FramesPerClip, self.stride, self.desired_width, self.desired_height)
        toc = time.clock()
        proc_time = toc-tic # for return or print
        print(VideoName+' Video hase taken ',proc_time,'seconds to be loaded')
        return video, clips
    
    def get_data(video_id):
        tic=time.clock()
        VideoName = self.VideoNames[video_id][0][0][0]
        DataName = VideoName+'/'+'Data.mat'
        DataMat = hdf5storage.loadmat(DataName)
        arraysD={}
        for k, v in DataMat.items():
            arraysD[k] = np.array(v)
        Fixations = arraysD[k]
        toc=time.clock()
        proc_time=toc-tic
        print(VideoName+' fixation data hase taken ',proc_time,'seconds to be loaded')
        return Fixations['fixdata']
