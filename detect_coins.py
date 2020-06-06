

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


import math
import numpy as np
import argparse
import glob
import cv2
import serial
import time
from twilio.rest import Client

#for privacy purposes I haven't given my account sid and token
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'

client = Client(account_sid, auth_token)


ser = serial.Serial('/dev/ttyACM0', 9600) #connecting to Arduino via Serial Communication


cap = cv2.VideoCapture(0)
def calculate_Histogram(img):
     
    m = np.zeros(img.shape[:2], dtype="uint8")
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.circle(m, (w, h), 60, 255, -1)

    
    h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        
    return cv2.normalize(h, h).flatten()

def calcHistogramFromFile(file):
    img = cv2.imread(file)
    return calculate_Histogram(img)
class Enum(tuple): __getattr__ = tuple.index


Coin = Enum(('Cent50', 'OneDollar', 'Cent5', 'Cent10'))

  
sample_images_50Cent = glob.glob("SampleImages/50Cent/*")
sample_images_OneDollar = glob.glob("SampleImages/OneDollar/*")
sample_images_5Cent = glob.glob("SampleImages/5Cent/*")
sample_images_10Cent = glob.glob("SampleImages/10Cent/*")

X = []
y = []


for i in sample_images_50Cent:
    X.append(calcHistogramFromFile(i))
    y.append(Coin.Cent50)
for i in sample_images_OneDollar:
    X.append(calcHistogramFromFile(i))
    y.append(Coin.OneDollar)
for i in sample_images_5Cent:
    X.append(calcHistogramFromFile(i))
    y.append(Coin.Cent5)
for i in sample_images_10Cent:
    X.append(calcHistogramFromFile(i))
    y.append(Coin.Cent10)


    
classif = MLPClassifier(solver="lbfgs")

   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

   
classiff.fit(X_train, y_train)

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    
    d = 1024 / image.shape[1]
    dim = (1024, int(image.shape[0] * d))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    output = image.copy()

    blurred_image = cv2.GaussianBlur(gray, (7, 7), 0)

    circles_identified = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
                               param1=200, param2=100, minRadius=50, maxRadius=120)


    def predictCoin(var):
       
        hist = calculate_Histogram(var)

        
        var2 = classif.predict([hist])

        
        return Coin[int(var2)]


    
    diameter = []
    coins = []
    coordinates = []

    count = 0
    if circles_identified is not None:
        
        for (x, y, r) in circles_identified[0, :]:
            diameter.append(r)

        
        circles_identified = np.round(circles_identified[0, :]).astype("int")

       
        for (x, y, d) in circles_identified:
            count += 1

            
            coordinates.append((x, y))

            
            roi = image[y - d:y + d, x - d:x + d]

            
            material = predictCoin(roi)
            coins.append(material)

            
            if False:
                m = np.zeros(roi.shape[:2], dtype="uint8")
                w = int(roi.shape[1] / 2)
                h = int(roi.shape[0] / 2)
                cv2.circle(m, (w, h), d, (255), -1)
                maskedCoin = cv2.bitwise_and(roi, roi, mask=m)
                cv2.imwrite("extracted/01coin{}.png".format(count), maskedCoin)

            
            cv2.circle(output, (x, y), d, (0, 255, 0), 2)
            cv2.putText(output, material,
                        (x - 40, y), cv2.FONT_HERSHEY_PLAIN,
                        1.5, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

   
    biggest = max(diameter)
    i = diameter.index(biggest)

    # scale everything according to maximum diameter
    # todo: this should be chosen by the user
    if coins[i] == "Cent50":
        diameter = [x / biggest * 35.25 for x in diameter]
        
    elif coins[i] == "OneDollar":
        diameter = [x / biggest * 25.25 for x in diameter]
        
    elif coins[i] == "Cent5":
        diameter = [x / biggest * 18.25 for x in diameter]
        
    elif coins[i] == "Cent10":
        diameter = [x / biggest * 22.25 for x in diameter]
        

   

    i = 0
    total = 0
    while i < len(diameter):
        d = diameter[i]
        m = coins[i]
        (x, y) = coordinates[i]
        t = "Unknown"

       
        if math.isclose(d, 35.25, abs_tol=3.5) and m == "Cent50":
            t = "50 Cent"
            total += 50
            message = client.messages.create( body='50 Cent Coin Detected!',from_='whatsapp:+14155238886', to='whatsapp:your_number')
            print(message.sid)
            ser.write(b"50Cent\n")
    
        elif math.isclose(d, 24.25, abs_tol=1.25) and m == "OneDollar":
            t = "One Dollar"
            total += 100
            message = client.messages.create( body='One Dollar Coin Detected!',from_='whatsapp:+14155238886', to='whatsapp:your_number')
            print(message.sid)
            ser.write(b"OneDolar\n")
            
        elif math.isclose(d, 18.25, abs_tol=2.5) and m == "Cent5":
            t = "5 Cent"
            total += 5
            message = client.messages.create( body='5 Cent Coin Detected!',from_='whatsapp:+14155238886', to='whatsapp:your_number')
            print(message.sid)
            ser.write(b"5Cent\n")
            
        elif math.isclose(d, 22.25, abs_tol=2.5) and m == "Cent10":
            t = "10 Cent"
            total += 10
            message = client.messages.create( body='10 Cent Coin Detected!',from_='whatsapp:+14155238886', to='whatsapp:your_number')
            print(message.sid)
            ser.write(b"10Cent\n")
        
 
    cv2.waitKey(0)
    
cap.release()
