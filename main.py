import cv2 as cv
import numpy as np
import os
import pyautogui as pg
from time import time
from PIL import ImageGrab
from windowcapture import *
import win32gui, win32ui, win32con
from vision import Vision
from hsvfilter import HsvFilter
from threading import Thread

store_win_cap = WindowCapture('Bluestacks')

# initializing the vision class
vision_ore = Vision('whirlpool.jpg')
is_bot_in_action = False


# prints out all open windows on computer
# from stackoverflow
def winEnumHandler(hwnd, ctx):
    if win32gui.IsWindowVisible(hwnd):
        print(hex(hwnd), win32gui.GetWindowText(hwnd))


def bot_actions(rectangles):
    if len(rectangles) > 0:
        targets = vision_ore.get_click_points(rectangles)
        target = store_win_cap.get_screen_position(targets[0])
        pg.moveTo(x=target[0], y=target[1])
        pg.click()
        pg.sleep(2)


loop_time = time()

# def winEnumHandler(hwnd, ctx):
#     if win32gui.IsWindowVisible(hwnd):
#         print(hex(hwnd), win32gui.GetWindowText(hwnd))
# win32gui.EnumWindows( winEnumHandler, None )

while True:
    screenshot = store_win_cap.get_screenshot()
    # does the object detection
    rectangles = vision_ore.find(screenshot, 0.7)

    # draw the detection results onto original img
    output_image = vision_ore.draw_rectangles(screenshot, rectangles)

    # display the processed img
    cv.imshow('Runescape Detection', output_image)
    if not is_bot_in_action:
        is_bot_in_action = True
        t = Thread(target=bot_actions, args=(rectangles,))
        t.start()

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print("Done")
