import cv2
import numpy as np

def get_transformed_points(kotak, sudut):
    poinbawah = []
    for petak in kotak:
        poins=np.array([[[int(petak[0]+(petak[2]*0.5)),int(petak[1]+petak[3])]]],dtype="float32")
        poin=cv2.perspectiveTransform(poins,sudut)[0][0]
        titik=[int(poin[0]),int(poin[1])]
        poinbawah.append(titik)
    return poinbawah

def cal_dis(a, b, jarakw, jarakh):
    h=abs(b[1]-a[1])
    w=abs(b[0]-a[0])
    jarw=float((w/jarakw)*180)
    jarh=float((h/jarakh)*180)
    return int(np.sqrt(((jarh)**2)+((jarw)**2)))

def get_distances(kotak1, poinbawah, jarakw, jarakh):
    jarakmat = []
    ktk = []
    for i in range(len(poinbawah)):
        for j in range(len(poinbawah)):
            if i!=j:
                jar=cal_dis(poinbawah[i], poinbawah[j], jarakw, jarakh)
                if jar<=150:
                    kedekatan=0
                    jarakmat.append([poinbawah[i], poinbawah[j], kedekatan])
                    ktk.append([kotak1[i], kotak1[j], kedekatan])
                elif jar>150 and jar<=180:
                    kedekatan=1
                    jarakmat.append([poinbawah[i], poinbawah[j], kedekatan])
                    ktk.append([kotak1[i], kotak1[j], kedekatan])
                else:
                    kedekatan=2
                    jarakmat.append([poinbawah[i], poinbawah[j], kedekatan])
                    ktk.append([kotak1[i], kotak1[j], kedekatan])
    return jarakmat, ktk

def get_count(distances_mat):
    red=[]
    yellow=[]
    green=[]
    for i in range(len(distances_mat)):
        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in red) and (distances_mat[i][0] not in green) and (distances_mat[i][0] not in yellow):
                red.append(distances_mat[i][0])
            if (distances_mat[i][1] not in red) and (distances_mat[i][1] not in green) and (distances_mat[i][1] not in yellow):
                red.append(distances_mat[i][1])
    for i in range(len(distances_mat)):
        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in red) and (distances_mat[i][0] not in green) and (distances_mat[i][0] not in yellow):
                yellow.append(distances_mat[i][0])
            if (distances_mat[i][1] not in red) and (distances_mat[i][1] not in green) and (distances_mat[i][1] not in yellow):
                yellow.append(distances_mat[i][1])
    for i in range(len(distances_mat)):
        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in red) and (distances_mat[i][0] not in green) and (distances_mat[i][0] not in yellow):
                green.append(distances_mat[i][0])
            if (distances_mat[i][1] not in red) and (distances_mat[i][1] not in green) and (distances_mat[i][1] not in yellow):
                green.append(distances_mat[i][1])
    return (len(red),len(yellow),len(green))