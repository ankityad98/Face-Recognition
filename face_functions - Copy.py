import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV,KFold
import pickle

label=''
def detect_face(frame):
    detector = cv2.CascadeClassifier("xml/frontal_face.xml")
    faces = detector.detectMultiScale(frame,1.2)
    return faces

def gray_scale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def cut_faces(image, faces_coord):
    faces = []
    for (x, y, w, h) in faces_coord:
        faces.append(image[y: y + h, x : x + w ])
    return faces

def normalize_intensity(images):
    images_norm = []
    for image in images:
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images,size=(47,62)):
    image_resize = []
    
    for image in images:
        img_size = cv2.resize(image,size)
        
        image_resize.append(img_size)
        
    return image_resize


def normalize_faces(frame, faces_coord):
    #gray_frame = gray_scale(frame)
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        cv2.rectangle(image, (x , y), (x + w , y + h), (0,0,255),2)


#Train model
def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
   
    people = [person for person in os.listdir("user/")]
   
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("user/" + person):
            if image.endswith('.jpg'):
                images.append(cv2.imread("user/" + person + '/' + image, 0))
                labels.append(i)
    return (images, np.array(labels), labels_dic)

def train_model():
    images, labels, labels_dic = collect_dataset()
    X_train=np.asarray(images)
    train=X_train.reshape(len(X_train),-1)
    
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(train.astype(np.float64))
    
    pca1 = PCA(n_components=.97)
    new_train=pca1.fit_transform(X_train_sc)
    kf=KFold(n_splits=5,shuffle=True)
    param_grid = {'C':[.0001,.001,.01,.1,1,10]}
    gs_svc = GridSearchCV(SVC(kernel='linear',probability=True),param_grid=param_grid,cv=kf,scoring='accuracy')

    gs_svc.fit(new_train,labels)
    svc1=gs_svc.best_estimator_
    
    filename = 'svc_linear_face.pkl'
    f=open(filename, 'wb')
    pickle.dump(svc1,f)
    f.close()

    filename = 'pca.pkl'
    f=open(filename, 'wb')
    pickle.dump(pca1,f)
    f.close()

    filename = 'standardscalar.pkl'
    f=open(filename, 'wb')
    pickle.dump(sc,f)
    f.close()
    print('model has been trained')
    return True

def predict():
    global label
    images, labels, labels_dic = collect_dataset()
    filename = 'svc_linear_face.pkl'
    svc1 = pickle.load(open(filename, 'rb'))

    filename = 'pca.pkl'
    pca1 = pickle.load(open(filename, 'rb'))

    filename = 'standardscalar.pkl'
    sc = pickle.load(open(filename, 'rb'))

    cam = cv2.VideoCapture(0)
    font=cv2.FONT_HERSHEY_PLAIN
    cv2.namedWindow("opencv_face", cv2.WINDOW_AUTOSIZE)



    while True:
        ret,frame = cam.read()
    
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces_coord = detect_face(gray) # detect more than one face
        if len(faces_coord):
            faces = normalize_faces(gray, faces_coord)
        
            for i, face in enumerate(faces): # for each detected face
                t=face.reshape(1,-1)
                t=sc.transform(t.astype(np.float64))
                test = pca1.transform(t)    
            
                prob=svc1.predict_proba(test)
                confidence = svc1.decision_function(test)
                print (confidence)
                print (prob)
           
            
            
                pred = svc1.predict(test)
                #print (pred,pred[0])
           
                name=labels_dic[pred[0]].capitalize()
                #print (name)
                
                if prob[0][1]>.85:
                
                    cv2.putText(frame, 'unknown',(faces_coord[i][0], faces_coord[i][1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (66, 53, 243), 2)
                
                
                elif prob[0][0]>.80:
                    cv2.putText(frame,'Ankit Yadav',(faces_coord[i][0], faces_coord[i][1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
                
                
                 
            
            draw_rectangle(frame, faces_coord) # rectangle around face
        
        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,cv2.LINE_AA)
    
        cv2.imshow("opencv_face", frame) # live feed in external
        if cv2.waitKey(5) == 27:
            break
        
    cam.release()
    cv2.destroyAllWindows()
        
