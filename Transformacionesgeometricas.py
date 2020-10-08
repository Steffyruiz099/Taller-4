import numpy as np
import os
import cv2
import math

dato = []
def click_and_crop(event, x, y, flags, param):
    global refPt, dato
    if event == cv2.EVENT_LBUTTONDOWN:
        dato.append((x,y))

# construct the argument parser and parse the arguments
path1 = input('ingrese la ruta de la imagen')            # se pide que ingrese la ruta de la imagen
image_name1 = input('ingrese el nombre de la imagen')    # se pide que ingrese el nombre de la imagen
path_file1 = os.path.join(path1, image_name1)              # se crea la ruta de la imagen
image1 = cv2.imread(path_file1)

path2 = input('ingrese la ruta de la imagen')  # se pide que ingrese la ruta de la imagen
image_name2 = input('ingrese el nombre de la imagen')  # se pide que ingrese el nombre de la imagen
path_file2 = os.path.join(path2, image_name2)  # se crea la ruta de la imagen
image2 = cv2.imread(path_file2)

clone = image1.copy()
cv2.namedWindow("image1")
cv2.setMouseCallback("image1", click_and_crop)
while True:
    cv2.imshow("image1", image1)
    cv2.waitKey(1) & 0xFF
    if len(dato) == 3:
        cv2.destroyAllWindows()
        print(dato)
        imagen2 = 1
        break

if imagen2 == 1:
    dato2 = []

    def click_and_crop2(event2, x2, y2, flags, param):
        global refPt2, dato2
        if event2 == cv2.EVENT_LBUTTONDOWN:
            dato2.append((x2, y2))

    clone2 = image2.copy()
    cv2.namedWindow("image2")
    cv2.setMouseCallback("image2", click_and_crop2)
    while True:
        cv2.imshow("image2", image2)
        cv2.waitKey(1) & 0xFF
        if len(dato2) == 3:
            cv2.destroyAllWindows()
            print(dato2)
            break

# affine
pts1 = np.float32([dato[0], dato[1], dato[2]])
pts2 = np.float32([dato2[0], dato2[1], dato2[2]])
M_affine = cv2.getAffineTransform(pts1, pts2)
image_affine = cv2.warpAffine(image1, M_affine, image1.shape[:2])

sx =  math.sqrt(np.power(M_affine[0][0], 2) + np.power(M_affine[1][0], 2))
sy =  math.sqrt(np.power(M_affine[0][1], 2) + np.power(M_affine[1][1], 2))
theta = -math.atan(M_affine[1][0]/M_affine[0][0])
theta_rad = theta * np.pi / 180
tx = ((M_affine[0][2] * math.cos(theta))-(M_affine[1][2] * math.sin(theta)))/sx
ty = ((M_affine[1][2] * math.cos(theta))+(M_affine[0][2] * math.sin(theta)))/sy

# similarity
M_sim = np.float32([[sx * np.cos(theta), -np.sin(theta_rad),tx],[np.sin(theta_rad), sy * np.cos(theta_rad), ty]])
image_similarity = cv2.warpAffine(image1, M_sim, image1.shape[:2])

print(M_affine)
print(M_sim)

error = image2 - image_similarity

errorcuadratico_median = math.sqrt(np.square(np.subtract(image2, image_similarity)).mean())
print(errorcuadratico_median)


cv2.imshow("Image1", image1)
cv2.imshow("Image2", image2)
cv2.imshow("Image affine", image_affine)
cv2.imshow("Image similarity", image_similarity)
cv2.imshow("error", error)
cv2.waitKey(0)

