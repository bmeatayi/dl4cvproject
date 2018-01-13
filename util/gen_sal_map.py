from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np

def _gen_meshgrid(x_min=0, x_max=1, y_min=0, y_max=1, res=0.01):

    '''
    input:
        - x_min, x_max: minimum and maximum x values
        - y_min, y_max: minimum and maximum y values
        - res: step size

    Output:
        - meshgrid
    '''

    x = np.arange(x_min, x_max + res, res)
    y = np.arange(y_min, y_max + res, res)
    X, Y = np.meshgrid(x, y)
    
    return X, Y


def _gen_gauss(fix_point, cov=((.005, 0), (0, .005))):

    '''
    Input:
        - fix_point: a single fixation point

    Output:
        - A 2D gaussian around the fixation point
    '''

    mean = fix_point
    cov = cov
    var = multivariate_normal(mean=mean, cov=cov)
    
    X, Y = _gen_meshgrid()
    
    Z = np.array([var.pdf([elx, ely]) for xv, yv in zip(X, Y) for elx, ely in zip(xv, yv)]).reshape(X.shape)
    Z /= np.max(Z)

    return Z


def frame_sal_map(fix_coor):
    '''
    Input:
        - fix_coor: is a (S x 2) numpy array.
            - F: number of Frames (or clips)
            - S: number of subjects
            
        * the x, y values are normalized between 0 and 1
        
    Output:
        - sal_map: is a (H x W)
    '''
    
    # initialize an empty saliency map
    sal_map = np.zeros(_gen_gauss(np.array([0,0])).shape)
    
    for subject in fix_coor:
        sal_map += _gen_gauss(subject) / fix_coor.shape[0]
    
    return sal_map


def vid_sal_map(fix_coor): # has the frame in first dimension

    '''
    Input:
        - fix_coor: is a (F x S x 2) numpy array.
            - F: number of Frames (or clips)
            - S: number of subjects
            
        * the x, y values are normalized between 0 and 1
        
    Output:
        - sal_map: is a (F x H x W)
    '''
    # initialize an empty saliency maps over frame
    F = fix_coor.shape[0]
    sal_map = np.zeros((F, _gen_gauss(np.array([0, 0])).shape[0], _gen_gauss(np.array([0, 0])).shape[1]))

    print('Computing saliency map for', F, 'frame..')

    for ind, frame in enumerate(fix_coor):
        sal_map[ind, :, :] = frame_sal_map(frame)
        print('Frame', ind, 'is done')

    print('=== Sliency map is computed for all frames ===')
    return sal_map


if __name__ == '__main__':

    fixation_data = np.random.rand(3,30,2)
    Z = vid_sal_map(fixation_data)

    for i in Z:
        plt.pcolor(i[:, :], cmap='gray')
        plt.axes().set_aspect('equal')
        plt.show()
