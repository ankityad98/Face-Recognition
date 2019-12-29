import os
import random
from scipy import ndarray
import cv2
# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]



def generate_images(fname='vineet',num_files_desired = 10):
    


    folder_path = 'data/'+fname
    #num_files_desired = 10

    # find all files paths from the folder
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # dictionary of the transformations we defined earlier
    available_transformations = {'rotate': random_rotation,'noise': random_noise,'horizontal_flip': horizontal_flip}


    num_generated_files = 0
    while num_generated_files <= num_files_desired:
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels
        #image_to_transform = sk.io.imread(image_path)
        img = cv2.imread(image_path,0)
        image_to_transform = np.array(img,dtype=np.float64)
        # random num of transformation to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))

        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1

        new_file = 'image_%d.jpg'%(len(images)+1)

        # write image to the disk
        #io.imsave(new_file_path, transformed_image)
        cv2.imwrite(folder_path+'/'+new_file,transformed_image)
        num_generated_files += 1

if __name__=='__main__':
    generate(40)
