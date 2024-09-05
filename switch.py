from gpiozero import Button
from signal import pause

SWITCH_PIN = 17  

button = Button(SWITCH_PIN)

def on_button_pressed():
    print("스위치가 눌렸습니다!")

def on_button_released():
    print("스위치가 떼어졌습니다!")

button.when_pressed = on_button_pressed
button.when_released = on_button_released

print("스위치 상태를 모니터링 중입니다. 종료하려면 Ctrl+C를 누르세요.")
pause()
