from __future__ import print_function
from __future__ import division
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import config
import microphone
import dsp
import led
import sys

visualization_type = sys.argv[1]
scroll_divisor_config = 4 if sys.argv[1] == "scroll_quad" else 2

_time_prev = time.time() * 1000.0
"""La última vez que la función frames_per_second() fue llamada"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
"""El filtro pasa-bajas es utilizado para estimar frames-per-second"""


def frames_per_second():
    """Devuelve los fotogramas por segundo estimados
    Devuelve la estimación actual de fotogramas por segundo (FPS).
    Los FPS se estiman midiendo la cantidad de tiempo que ha transcurrido desde que
    esta función fue llamada previamente. La estimación de FPS es filtrada con el pasa-bajas
    para reducir el ruido.
    Esta función está diseñada para ser llamada una vez por cada iteración de
    bucle principal del programa.
    Returns
    -------
    fps : float
        frames-per-second estimados. Este valor es filtrado con el pasa-bajas
        para reducir el ruido.
    """
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


def memoize(function):
    """Proporciona un decorador para memorizar funciones."""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper


@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    """Cambia el tamaño de la matriz de forma inteligente interpolando linealmente los valores
    Parámetros
    ----------
    y : np.array
        Matriz que debería cambiar de tamaño
    new_length : int
        La longitud de la nueva matriz interpolada
    Returns
    -------
    z : np.array
        Nueva matriz con una longitud new_length que contiene valores interpolados
        de y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z


r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.2, alpha_rise=0.99)
g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.05, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.1, alpha_rise=0.5)
common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.99, alpha_rise=0.01)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)),
                       alpha_decay=0.1, alpha_rise=0.99)
# scroll_divisor_config config is set to 2 if scroll_quad is sent in the arg
p = np.tile(1.0, (3, config.N_PIXELS // scroll_divisor_config))
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)


def visualize_scroll(y):
    """Efecto que se origina en el centro y se desplaza hacia las orillas"""
    global p
    y = y**2.0
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))
    # Ventana de efecto de desplazamiento
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)
    # Crea un nuevo color que se origina en el centro
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    # Actualiza el LED ring
    return np.concatenate((p[:, ::-1], p), axis=1)


def visualize_scroll_quad(y):
    """Efecto que se origina en 2 puntos centrales y se desplaza hacia afuera
    Solo debe usarse con un número de LED divisible por 4
    """
    global p
    y = y**2.0
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))
    # Ventana de efecto de desplazamiento
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)
    # Crea un nuevo colororiginándolo en el centro
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    # Actualiza el LED ring
    return np.concatenate((p[:, ::-1], p, p[:, ::-1], p), axis=1)


def visualize_scroll_in(y):
    """Efecto que se origina en la orilla y se desplaza hacia el centro"""
    global p
    y = y**2.0
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))
    # Ventana de efecto de desplazamiento
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)
    # Crea un nuevo color originándolo en el centro
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    # Actualiza el LED ring
    return np.concatenate((p[:, :], p[:, ::-1]), axis=1)


def visualize_energy(y):
    """Efecto que se expande desde el centro con el incremento de la energía del sonido"""
    global p
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Escala para el ancho del LED ring
    y *= float((config.N_PIXELS // 2) - 1)
    # Mapa de canales de color según la energía en las diferentes bandas de frecuencia.
    scale = 0.9
    r = int(np.mean(y[:len(y) // 3]**scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(np.mean(y[2 * len(y) // 3:]**scale))
    # Asigna colores a diferentes regiones de frecuencia
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Aplicar un desenfoque sustancial para suavizar los bordes.
    p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=4.0)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=4.0)
    # Establece el nuevo valor del pixel
    return np.concatenate((p[:, ::-1], p), axis=1)


_prev_spectrum = np.tile(0.01, config.N_PIXELS // 2)


def visualize_spectrum(y):
    """Efecto que mapea las frecuencias del banco de filtros MEL en la tira de LED"""
    global _prev_spectrum
    y = np.copy(interpolate(y, config.N_PIXELS // 2))
    common_mode.update(y)
    diff = y - _prev_spectrum
    _prev_spectrum = np.copy(y)
    # Asignaciones de canales de color
    r = r_filt.update(y - common_mode.value)
    g = np.abs(diff)
    b = b_filt.update(np.copy(y))
    # Refleja los canales de color para una salida simétrica
    r = np.concatenate((r[::-1], r))
    g = np.concatenate((g[::-1], g))
    b = np.concatenate((b[::-1], b))
    output = np.array([r, g,b]) * 255
    return output


fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD,
                       alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()


def microphone_update(audio_samples):
    global y_roll, prev_rms, prev_exp, prev_fps_update
    # Normaliza muestras entre 0 y 1
    y = audio_samples / 2.0**15
    # Construye una ventana continua de muestras de audio
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)
    
    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        print('No audio input. Volume below threshold. Volume:', vol)
        led.pixels = np.tile(0, (3, config.N_PIXELS))
        led.update()
    else:
        # Transforma la entrada de audio en el dominio de frecuencia
        N = len(y_data)
        N_zeros = 2**int(np.ceil(np.log2(N))) - N
        # Rellenar con ceros hasta la próxima potencia de dos
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
        YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
        # Construye un banco de filtros MEL a partir de los datos de FFT
        mel = np.atleast_2d(YS).T * dsp.mel_y.T
        # Escala los datos a valores más adecuados para la visualización
        # mel = np.sum(mel, axis=0)
        mel = np.sum(mel, axis=0)
        mel = mel**2.0
        # Obtiene la normalización
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)
        # Asigna la salida del banco de filtros al LED ring
        output = visualization_effect(mel)
        led.pixels = output
        led.update()
        if config.USE_GUI:
            # Traza la salida del banco de filtros
            x = np.linspace(config.MIN_FREQUENCY, config.MAX_FREQUENCY, len(mel))
            mel_curve.setData(x=x, y=fft_plot_filter.update(mel))
            # Traza los canales de color
            r_curve.setData(y=led.pixels[0])
            g_curve.setData(y=led.pixels[1])
            b_curve.setData(y=led.pixels[2])
    if config.USE_GUI:
        app.processEvents()
    
    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if time.time() - 0.5 > prev_fps_update:
            prev_fps_update = time.time()
            print('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))


# Número de muestras de audio para leer cada cuadro de tiempo
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Matriz que contiene la ventana de muestra continua de audio.
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16

if sys.argv[1] == "spectrum":
        visualization_type = visualize_spectrum
elif sys.argv[1] == "energy":
        visualization_type = visualize_energy
elif sys.argv[1] == "scroll":
        visualization_type = visualize_scroll
elif sys.argv[1] == "scroll_in":
        visualization_type = visualize_scroll_in
elif sys.argv[1] == "scroll_quad":
        visualization_type = visualize_scroll_quad
else:
        visualization_type = visualize_spectrum

visualization_effect = visualization_type
"""Efecto de visualización para mostrar en el LED ring"""


if __name__ == '__main__':
    if config.USE_GUI:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtGui, QtCore
        # Crea la ventana GUI
        app = QtGui.QApplication([])
        view = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(100,100,100))
        view.setCentralItem(layout)
        view.show()
        view.setWindowTitle('Visualization')
        view.resize(800,600)
        # Gráfico del banco de filtros MEL
        fft_plot = layout.addPlot(title='Filterbank Output', colspan=3)
        fft_plot.setRange(yRange=[-0.1, 1.2])
        fft_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        x_data = np.array(range(1, config.N_FFT_BINS + 1))
        mel_curve = pg.PlotCurveItem()
        mel_curve.setData(x=x_data, y=x_data*0)
        fft_plot.addItem(mel_curve)
        # Gráfico de visualización
        layout.nextRow()
        led_plot = layout.addPlot(title='Visualization Output', colspan=3)
        led_plot.setRange(yRange=[-5, 260])
        led_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        # Lápiz para cada una de las curvas de canal de color
        r_pen = pg.mkPen((255, 30, 30, 200), width=4)
        g_pen = pg.mkPen((30, 255, 30, 200), width=4)
        b_pen = pg.mkPen((30, 30, 255, 200), width=4)
        # Canal de curvas de color
        r_curve = pg.PlotCurveItem(pen=r_pen)
        g_curve = pg.PlotCurveItem(pen=g_pen)
        b_curve = pg.PlotCurveItem(pen=b_pen)
        # Define x datos
        x_data = np.array(range(1, config.N_PIXELS + 1))
        r_curve.setData(x=x_data, y=x_data*0)
        g_curve.setData(x=x_data, y=x_data*0)
        b_curve.setData(x=x_data, y=x_data*0)
        # Agrega curvas para el gráfico
        led_plot.addItem(r_curve)
        led_plot.addItem(g_curve)
        led_plot.addItem(b_curve)
        # Etiqueta de rango de frecuencia
        freq_label = pg.LabelItem('')
        # Control deslizante de frecuencia
        def freq_slider_change(tick):
            minf = freq_slider.tickValue(0)**2.0 * (config.MIC_RATE / 2.0)
            maxf = freq_slider.tickValue(1)**2.0 * (config.MIC_RATE / 2.0)
            t = 'Frequency range: {:.0f} - {:.0f} Hz'.format(minf, maxf)
            freq_label.setText(t)
            config.MIN_FREQUENCY = minf
            config.MAX_FREQUENCY = maxf
            dsp.create_mel_bank()
        freq_slider = pg.TickSliderItem(orientation='bottom', allowAdd=False)
        freq_slider.addTick((config.MIN_FREQUENCY / (config.MIC_RATE / 2.0))**0.5)
        freq_slider.addTick((config.MAX_FREQUENCY / (config.MIC_RATE / 2.0))**0.5)
        freq_slider.tickMoveFinished = freq_slider_change
        freq_label.setText('Frequency range: {} - {} Hz'.format(
            config.MIN_FREQUENCY,
            config.MAX_FREQUENCY))
        # Selección de efectos
        active_color = '#16dbeb'
        inactive_color = '#FFFFFF'
        def energy_click(x):
            global visualization_effect
            visualization_effect = visualize_energy
            energy_label.setText('Energy', color=active_color)
            scroll_label.setText('Scroll', color=inactive_color)
            spectrum_label.setText('Spectrum', color=inactive_color)
        def scroll_click(x):
            global visualization_effect
            visualization_effect = visualize_scroll
            energy_label.setText('Energy', color=inactive_color)
            scroll_label.setText('Scroll', color=active_color)
            spectrum_label.setText('Spectrum', color=inactive_color)
        def spectrum_click(x):
            global visualization_effect
            visualization_effect = visualize_spectrum
            energy_label.setText('Energy', color=inactive_color)
            scroll_label.setText('Scroll', color=inactive_color)
            spectrum_label.setText('Spectrum', color=active_color)
        # Crea "botones" de efectos (etiquetas con evento de clic)
        energy_label = pg.LabelItem('Energy')
        scroll_label = pg.LabelItem('Scroll')
        spectrum_label = pg.LabelItem('Spectrum')
        energy_label.mousePressEvent = energy_click
        scroll_label.mousePressEvent = scroll_click
        spectrum_label.mousePressEvent = spectrum_click
        energy_click(0)
        # Diseño
        layout.nextRow()
        layout.addItem(freq_label, colspan=3)
        layout.nextRow()
        layout.addItem(freq_slider, colspan=3)
        layout.nextRow()
        layout.addItem(energy_label)
        layout.addItem(scroll_label)
        layout.addItem(spectrum_label)
    # Inicializa los LEDs
    led.update()
    # Empieza a escuchar la transmisión de audio en vivo
    microphone.start_stream(microphone_update)