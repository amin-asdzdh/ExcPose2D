import cv2
import numpy as np

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class Rotate_90_CC(object):
    """Rotate the image and keypoints 90 degrees counter-clockwise.
    
    """

    def __init__(self, rotate_angle = 90):
        self.rotate_angle = rotate_angle

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        im_center = (w/2, h/2)
        
        # rotate image
        
        rotation_mat = cv2.getRotationMatrix2D(im_center, 90, 1.)
    
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])
    
        # find the new width and height bounds
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
    
        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - im_center[0]
        rotation_mat[1, 2] += bound_h/2 - im_center[1]
    
        # rotate image with the new bounds and translated rotation matrix
        rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
        
        # rotate keypoints too
        key_pts_t = key_pts.copy()
        for j in range(len(key_pts)):
            keypoint = key_pts[j, :]
            key_pts_t[j, 0] = keypoint[1]
            key_pts_t[j, 1] = w - keypoint[0]
        
        return {'image': rotated_image, 'keypoints': key_pts_t}
    
    def _rotatePoint(self, pt, angle, image_shape):
        
        c_w = image_shape[1]
        c_h = image_shape[0]
        
        a = np.radians(angle)
        cosa = np.cos(a)
        sina = np.sin(a)
        
        # step1: translate the center of rotation to (0, 0)
        pt[0] = pt[0] - c_w/2
        pt[1] = pt[1] - c_h/2
        
        # step2: rotate the point about (0, 0)
        pt_t = [pt[0]*cosa - pt[1]*sina, pt[0] * sina + pt[1] * cosa]
        
        # step3: translate back 
        pt_t[0] = pt_t[0] + c_w/2
        pt_t[1] = pt_t[1] + c_h/2
        
        return pt_t
    