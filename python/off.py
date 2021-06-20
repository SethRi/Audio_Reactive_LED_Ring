#!/usr/bin/env python3

import time
from rpi_ws281x import *
import argparse
import config

# Configuración del LED ring:
LED_COUNT = config.N_PIXELS # Number of LED pixels.
LED_PIN = config.LED_PIN  # GPIO pin connected to the pixels (18 uses PWM!).
LED_FREQ_HZ = config.LED_FREQ_HZ   # LED signal frequency in hertz (usually 800khz)
LED_DMA = config.LED_DMA  # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = config.BRIGHTNESS  # Set to 0 for darkest and 255 for brightest
LED_INVERT = config.LED_INVERT  # True to invert the signal (when using NPN transistor level shift)

# No en la configuración
LED_CHANNEL = 0  # Coloca '1' para los GPIOs 13, 19, 41, 45 or 53


# Definir funciones que animan los LEDs de varias formas.
def color_wipe(strip, color, wait_ms=50):
    """Limpia el color en la pantalla, un píxel a la vez."""
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
        strip.show()
        time.sleep(wait_ms / 1000.0)


# Continua el main:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clear', action='store_true', help='clear the display on exit')
    args = parser.parse_args()

    # Crea el objeto NeoPixel con la configuración apropiada.
    strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    # Inicializa la librería (debe ser llamada una vez antes de otras funciones).
    strip.begin()

    color_wipe(strip, Color(0, 0, 0), 10)