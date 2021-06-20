  
"""Cinfiguracion para audio reactive ring"""
from __future__ import print_function
from __future__ import division
import os

DEVICE = 'pi'


""" pi significa que utiizaremos Raspberry Pi como una unidad independiente para procesar
la entrada de audio y controlar el LED ring directamente."""


if DEVICE == 'pi':
    LED_PIN = 18
    """GPIO pin conectado a LED ring pixeles (debe soportar PWM)"""
    LED_FREQ_HZ = 800000
    """LED frecuencia de la señal en Hz (normalmente 800kHz)"""
    LED_DMA = 5
    """Canal DMA usado para generar señales PWM (intentar con 5)"""
    BRIGHTNESS = 255
    """Luminosidad del LED ring entre 0 y 255"""
    LED_INVERT = False
    """Colocar True si se usa un convertidor inverso de nivel lógico"""
    SOFTWARE_GAMMA_CORRECTION = True
    """Cambiar a True ya que Raspberry Pi no usa variaciones en hardware"""



USE_GUI = False
"""Mostrar o no un gráfico de visualización de la GUI de PyQtGraph"""

DISPLAY_FPS = True
"""Mostrar los FPS cuando esté funcionando (puede reducir el desempeño)"""

N_PIXELS = 24
"""Número de pixeles en el LED ring (debe ser múltiplo de 2)"""

GAMMA_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'gamma_table.npy')
"""Ubicación de la tabla de corrección de gamma"""

MIC_RATE = 44100  # 48000
"""Frecuencia de muestreo del micrófono en Hz"""

FPS = 50
"""Frecuencia de actualización deseada para la visualización (fotogramas por segundo),
FPS indica la frecuencia de actualización deseada o fotogramas por segundo de la visualización de audio.
La frecuencia de actualización real puede ser menor si la computadora no puede mantenerse
con el valor de FPS deseado.
Las tasas de fotogramas más altas mejoran la "capacidad de respuesta" y reducen la latencia de la
visualización pero demandan una mayor cantidad de recursos.
Las bajas velocidades de fotogramas requieren de menos recursos pero la visualización puede
parecer "lenta" o desincronizada con el audio que se está reproduciendo si es demasiada baja.
Los FPS no deben exceder la frecuencia de actualización máxima del LED ring, que
depende de la longitud del LED ring."""

_max_led_FPS = int(((N_PIXELS * 30e-6) + 50e-6)**-1.0)
assert FPS <= _max_led_FPS, 'FPS must be <= {}'.format(_max_led_FPS)

MIN_FREQUENCY = 200
"""Las frecuencias por debajo de este valor se eliminarán durante el procesamiento de audio."""

MAX_FREQUENCY = 12000
"""Las frecuencias por encima de este valor se eliminarán durante el procesamiento de audio."""

N_FFT_BINS = 24
"""Número de contenedores de frecuencia que se utilizarán al transformar el audio al dominio de frecuencia
Las transformadas rápidas de Fourier se utilizan para transformar datos de audio en el dominio del tiempo al
dominio de la frecuencia. Las frecuencias presentes en la señal de audio se asignan
a sus respectivos contenedores de frecuencia. Este valor indica el número de
contenedores de frecuencia a utilizar.
Un pequeño número de contenedores reduce la resolución de frecuencia de la visualización.
pero mejora la resolución de amplitud. Contrario a cuando se usa un gran
número de contenedores. 
No tiene sentido usar más contenedores que píxeles en el LED ring."""

N_ROLLING_HISTORY = 2
"""Número de fotogramas de audio anteriores para incluir en la ventana móvil"""

MIN_VOLUME_THRESHOLD = 1e-7
"""No se muestra visualización de música si el volumen de audio grabado está por debajo del umbral"""
