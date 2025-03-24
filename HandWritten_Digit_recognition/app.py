# -*- coding: utf-8 -*-

import pygame, sys
from pygame.locals import *
import numpy as np
import cv2  # Added import
from keras.models import load_model
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "bestmodel.h5")
MODEL = load_model(model_path)

WINDOWSIZEX = 640
WINDOWSIZEY = 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
IMAGESAVE = False

LABELS = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}
BOUNDARYINC = 5

pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 32)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1
PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:  # Check if lists are not empty
                rect_min_x = max(min(number_xcord) - BOUNDARYINC, 0)
                rect_max_x = min(max(number_xcord) + BOUNDARYINC, WINDOWSIZEX)
                rect_min_y = max(min(number_ycord) - BOUNDARYINC, 0)
                rect_max_y = min(max(number_ycord) + BOUNDARYINC, WINDOWSIZEY)
                number_xcord = []
                number_ycord = []
                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[
                    rect_min_x:rect_max_x, rect_min_y:rect_max_y
                ].T.astype(np.float32)

                if IMAGESAVE:
                    cv2.imwrite("image.png", img_arr)
                    image_cnt += 1
                if PREDICT:
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), "constant", constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255
                    label = str(
                        LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))]
                    )
                    text_surf = FONT.render(label, True, RED, WHITE)
                    text_rect = text_surf.get_rect()
                    text_rect.left, text_rect.bottom = rect_min_x, rect_max_y
                    DISPLAYSURF.blit(text_surf, text_rect)
        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
