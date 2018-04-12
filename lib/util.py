
import matplotlib.pyplot as plt
import cv2 as cv

def show_image(img):
    shown =  cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(shown)
    plt.xticks([]), plt.yticks([])
    plt.show()

def show_comparison_image(image1, image2):
    img1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(img1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(img2)
    plt.xticks([]), plt.yticks([])
    plt.show()
