from __future__ import print_function
from __future__ import division

import platform
import numpy as np
import config


# Raspberry Pi controla el LED ring directamente
elif config.DEVICE == 'pi':
    from rpi_ws281x import *
    strip = Adafruit_NeoPixel(config.N_PIXELS, config.LED_PIN,
                                       config.LED_FREQ_HZ, config.LED_DMA,
                                       config.LED_INVERT, config.BRIGHTNESS)
    strip.begin()
elif config.DEVICE == 'blinkstick':
    from blinkstick import blinkstick
    import signal
    import sys
    # Apagará todas las luces cuando es llamado.
    def signal_handler(signal, frame):
        all_off = [0]*(config.N_PIXELS*3)
        stick.set_led_data(0, all_off)
        sys.exit(0)

    stick = blinkstick.find_first()
    # Crea un listener que apaga las luces cuando el programa termina.
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

_gamma = np.load(config.GAMMA_TABLE_PATH)
"""Tabla de búsqueda de gamma utilizada para la corrección de brillo no lineal"""

_prev_pixels = np.tile(253, (3, config.N_PIXELS))
"""Valores de píxeles que se mostraron más recientemente en la tira de LED"""

pixels = np.tile(1, (3, config.N_PIXELS))
"""Valores de píxeles para el LED ring"""

_is_python_2 = int(platform.python_version_tuple()[0]) == 2


def _update_pi():
    """Raspberry escribe nuevos valores de LED en el LED ring.
    Raspberry Pi usa rpi_ws281x para controlar el LED ring directamente.
    Esta función actualiza el LED ring con nuevos valores."""

    global pixels, _prev_pixels
    # Trunca valores y los convierte en números enteros
    pixels = np.clip(pixels, 0, 255).astype(int)
    # Corrección opcional de la gamma
    p = _gamma[pixels] if config.SOFTWARE_GAMMA_CORRECTION else np.copy(pixels)
    # Codifica valores de LED de 24 bits en números enteros de 32 bits
    r = np.left_shift(p[0][:].astype(int), 8)
    g = np.left_shift(p[1][:].astype(int), 16)
    b = p[2][:].astype(int)
    rgb = np.bitwise_or(np.bitwise_or(r, g), b)
    # Actualiza los píxeles
    for i in range(config.N_PIXELS):
        # Ignora los píxeles si no han cambiado (ahorra ancho de banda)
        if np.array_equal(p[:, i], _prev_pixels[:, i]):
            continue
            
        strip._led_data[i] = int(rgb[i])
    _prev_pixels = np.copy(p)
    strip.show()

def _update_blinkstick():
    """Escribe nuevos valores de LED en el Blinkstick.
       Esta función actualiza el LED ring con nuevos valores."""

    global pixels
    
    # Trunca valores y los convierte en números enteros
    pixels = np.clip(pixels, 0, 255).astype(int)
    # Corrección opcional de la gamma
    p = _gamma[pixels] if config.SOFTWARE_GAMMA_CORRECTION else np.copy(pixels)
    # Lee los valores RGB
    r = p[0][:].astype(int)
    g = p[1][:].astype(int)
    b = p[2][:].astype(int)

    # Crea una matriz en la que almacenaremos los estados del led
    newstrip = [None]*(config.N_PIXELS*3)

    for i in range(config.N_PIXELS):
        # blinkstick usa el formato GRB
        newstrip[i*3] = g[i]
        newstrip[i*3+1] = r[i]
        newstrip[i*3+2] = b[i]
    # envia los datos a blinkstick
    stick.set_led_data(0, newstrip)


def update():
    """Actualiza los valores del LED ring"""
    if config.DEVICE == 'pi':
        _update_pi()
    elif config.DEVICE == 'blinkstick':
        _update_blinkstick()
    else:
        raise ValueError('Invalid device selected')


# Ejecuta este archivo para realizar una prueba de funcionamiento de los LEDs
# Si todo está funcionando, debería ver un desplazamiento de píxeles rojo, verde y azul
# a lo largo de la tira de LED continuamente
if __name__ == '__main__':
    import time
    # Apaga todos los pixeles
    pixels *= 0
    pixels[0, 0] = 255  # Establece el primer pixel rojo
    pixels[1, 1] = 255  # Establece el segundo pixel verde
    pixels[2, 2] = 255  # Establece el tercer pixel azul
    print('Starting LED strand test')
    while True:
        pixels = np.roll(pixels, 1, axis=1)
        update()
        time.sleep(.1)