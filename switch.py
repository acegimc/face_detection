from gpiozero import Button
from time import sleep

SWITCH_PIN = 27 
button = Button(SWITCH_PIN)

print("ctrl+C->exit")

while True:
    if button.is_pressed:
        print("On!")
        sleep(1)  
    sleep(1)
