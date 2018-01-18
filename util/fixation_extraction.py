import hdf5storage
import numpy as np

class FixationLoader():
    
    def __init__(self, *args, **kwargs):
        pass
    
    def get_datadotmat(self, filename):
        DataMat = hdf5storage.loadmat(filename)
        Data = DataMat['Data'][0]
        VideoName = Data[0][0][0]
        VideoFramerate = Data[1][0][0]
        VideoFrames = Data[2][0][0]
        VideoSize = Data[3][0]
        fixdata = Data[4]

        return VideoName, VideoFramerate, VideoFrames, VideoSize, fixdata
    
    def _sort_fixdata(self, data):
        data = np.array(sorted(data, key=lambda  l:l[0]))
        subject_ind = np.sort(np.array(list(set(data[:,0]))))
        for ind in subject_ind:
            data[data[:,0] == ind, 1:] = np.array(sorted(data[data[:,0] == ind, 1:], key=lambda  l:l[0]))

        return data
    
   
    def _get_frame_ind_range(self, start, duration, frameDistMS):
        
        '''
         # this function computes the index of the frame, given the startn duration and 
         the time period between two frames in a video
        
        '''
        return list(np.arange(int(np.floor(start/ frameDistMS)), int(np.floor((start + duration) / frameDistMS)) + 1))
    
    
    def _fix_fixation_subject(self, fixation_ind, fixation_coor):
        fx_coor = np.array(fixation_coor)
        fx_ind = np.array(fixation_ind)
        fx_coor_out = np.zeros((np.max(fixation_ind)+1, 2))

        for indx, ind in enumerate(fx_ind):
            fx_coor_out[ind] = fx_coor[indx]

        for indx, fixation in enumerate(fx_coor_out):
            if fixation[0] == 0 and fixation[1] == 0:
                fx_coor_out[indx] = fx_coor_out[indx-1]

        return fx_coor_out
    
    def _fix_fixation_video(self, fixation_ind, fixation_coor):
        final_fixation = []
        for subject in range(len(fixation_ind)):
            final_fixation.append(list(self._fix_fixation_subject(fixation_ind[subject], fixation_coor[subject])))

        final_fixation = np.array(final_fixation)

        return final_fixation
    
    def _get_fixation_ind_coor(self, fixdata_sorted, VideoFramerate):
    
        frameDistMS = (1/VideoFramerate) * 1000 # in ms
        unique_subject_ind = list(set(fixdata_sorted[:,0]))

        list_video_fix_coor = []
        list_video_fix_ind = []

        for subject in unique_subject_ind:
            subject_int = int(subject) - 1

            list_subject_fix = []    
            list_subject_fix_xy = []

            for fix_start, fix_dur, fix_x, fix_y in fixdata_sorted[fixdata_sorted[:,0]==subject,1:]:
                # get the frame index
                list_subject_fix += self._get_frame_ind_range(fix_start, fix_dur, frameDistMS)
                # get the fixation points
                list_subject_fix_x = [fix_x] * len(self._get_frame_ind_range(fix_start, fix_dur, frameDistMS))
                list_subject_fix_y = [fix_y] * len(self._get_frame_ind_range(fix_start, fix_dur, frameDistMS))
                list_subject_fix_xy += list(zip(list_subject_fix_x, list_subject_fix_y))

            list_video_fix_ind.append(list_subject_fix)
            list_video_fix_coor.append(list_subject_fix_xy)

        fixation_ind = np.array(list_video_fix_ind)
        fixation_coor = np.array(list_video_fix_coor)
        
        return fixation_ind, fixation_coor
    
    def _video_fixation(self, fixed_fixation, VideoFrames, VideoSize):
    
        final_fixation = np.zeros((int(VideoFrames), len(fixed_fixation), 2))  # frames x subject x 2
        
        for frame_ind, frame in enumerate(final_fixation):   
            for subject_ind, subject in enumerate(frame):
                if frame_ind < len(fixed_fixation[subject_ind]): 
                # this condition is because some subject do not have fixation as much as framesize
                    final_fixation[frame_ind, subject_ind] = fixed_fixation[subject_ind][frame_ind] / VideoSize
                
        return final_fixation
    
    
    def get_video_fixation(self, filename=None):
        
        _, VideoFramerate, VideoFrames, VideoSize, fixdata = self.get_datadotmat(filename)
        
        fixdata_sorted = self._sort_fixdata(fixdata)

        fixation_ind, fixation_coor = self._get_fixation_ind_coor(fixdata_sorted, VideoFramerate)
        # at this point we have the frame indeces and postion for each fixation, but they are incostent, so..

        fixed_fixation = self._fix_fixation_video(fixation_ind, fixation_coor)
        # now the fixation values are consistent with frame indeces. We need to get it in the format that we want:
        # Frames x Subject x

        final_fixation = self._video_fixation(fixed_fixation, VideoFrames, VideoSize)
        
        return final_fixation
		
		
if __name__ == '__main__':

	fix_obj = FixationLoader()
	fixation_data = fix_obj.get_video_fixation('Data.mat')
	print(fixation_data.shape)