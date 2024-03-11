# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:38:49 2023

@author: uzcheng
"""
import pyautogui as ui
import time
ui.FAILSAFE = False

# %%
while True:
    pos = ui.position()
    print(pos)
    time.sleep(0.5)
    if pos == (0,0): break

size = ui.size()
x, y, t, clicks, amount= 100, 100, 2, 3, 3000

# %% Move and Drag
ui.moveTo(x=x, y=y, duration=t)
ui.moveRel(x=x, y=y, duration=t)
ui.dragTo(x=x, y=y, duration=t)
ui.dragRel(x=x, y=y, duration=t)

# %% Clicks
ui.click(x=x, y=y, clicks=clicks, interval=t, button='left')
ui.click(x=x, y=y, clicks=clicks, interval=t, button='middle')
ui.click(x=x, y=y, clicks=clicks, interval=t, button='right')

ui.rightClick(x=x, y=y)
ui.middleClick(x=x, y=y)
ui.doubleClick(x=x, y=y)
ui.tripleClick(x=x, y=y)

ui.scroll(amount, x=x, y=y)

# %% Alerts
ui.alert('This displays some text with an OK button.')
ui.confirm('This displays text and has an OK and Cancel button.')
ui.prompt('This lets the user type in a string and press OK.')

# %% Hotkeys and keyboard
ui.hotkey('alt', 'tab')
ui.typewrite('Hello world!\n', interval=0.05)
