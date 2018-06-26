from utils import *
import cv2
img = cv2.cvtColor(cv2.imread('images/Vd-Orig.png'), cv2.COLOR_BGR2RGB)
plot_with_kernels(img)
