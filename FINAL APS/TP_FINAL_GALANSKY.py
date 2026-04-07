# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:12:03 2026

@author: Tatiana Galansky

TP FINAL APS
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pandas as pd
from scipy import signal as sig
from scipy.signal import find_peaks
import warnings
from pytc2.sistemas_lineales import plot_plantilla
warnings.filterwarnings('ignore')


# CONFIGURACIÓN

fs = 4000
DATA_PATH = 'circor_dataset'

# IDs de pacientes
IDS_SANOS = ['50048', '49998', '50247', '84687', '84688', '85336', '85091', 
             '84804', '84807', '84922']

IDS_SOPLO = ['49751', '50159', '50746', '50326', '85036', '55945', '68435', 
             '68374', '84690', '84732', '84857']

# IDs de pacientes específicos que voy a usar de ejemplo para los gráficos
ID_REPRESENTATIVO_SANO = '49998'
ID_REPRESENTATIVO_SOPLO = '50746'

# Para los títulos de los gráficos
PACIENTE_SANO = ID_REPRESENTATIVO_SANO
PACIENTE_SOPLO = ID_REPRESENTATIVO_SOPLO

#%% FUNCIONES DE PROCESAMIENTO DE SEÑALES

def cargar_señal_raw(p_id):
    """Carga la señal original sin filtrar"""
    file_path = os.path.join(DATA_PATH, f"{p_id}_MV")
    
    try:
        record = wfdb.rdrecord(file_path)
        señal = record.p_signal[:, 0]
        señal = señal / (np.max(np.abs(señal)) + 1e-12)
        return señal
    except Exception as e:
        print(f"Error en {p_id}: {e}")
        return None
    
def cargar_y_filtrar(p_id, fs=4000):
    file_path = os.path.join(DATA_PATH, f"{p_id}_MV")
    
    try:
        record = wfdb.rdrecord(file_path)
        señal = record.p_signal[:, 0]
        señal = señal / (np.max(np.abs(señal)) + 1e-12) 
        
        # --- 1. BUTTERWORTH (Diseño y Aplicación) ---
        wp = [20, 600]
        ws = [1, 650]
        sos_butter = sig.iirdesign(wp=wp, ws=ws, gpass=1, gstop=60,
                                   analog=False, ftype='butter', output='sos', fs=fs)
        s_butter = sig.sosfiltfilt(sos_butter, señal)
        
        # --- 2. FIR (Diseño y Aplicación) ---
        numtaps = 4001
        f_fir = [0, ws[0], wp[0], wp[1], ws[1], fs/2]
        g_fir = [0, 0, 1, 1, 0, 0]
        fir_coeffs = sig.firwin2(numtaps, freq=f_fir, gain=g_fir, window='boxcar', fs=fs)
        s_fir = sig.filtfilt(fir_coeffs, 1.0, señal)
        
        # --- 3. POST-PROCESAMIENTO ---
        def quitar_base(s):
            base = sig.medfilt(sig.medfilt(s, 101), 201)
            res = s - base
            return res / (np.max(np.abs(res)) + 1e-12)

        s_butter_final = quitar_base(s_butter)
        s_fir_final = quitar_base(s_fir)
        
        # DEVOLVEMOS: señales finales, señal original Y LOS COEFICIENTES
        return s_butter_final, s_fir_final, señal, sos_butter, fir_coeffs
        
    except Exception as e:
        print(f"Error en {p_id}: {e}")
        return [None]*5

# --- EJECUCIÓN Y GRÁFICOS ---
p_id_test = IDS_SANOS[1]

sb, sf, original, sos_b, fir_c = cargar_y_filtrar(p_id_test, fs=4000)
wp = [20, 600]
ws = [1, 650]
gpass=1 
gstop=60

if sb is not None:
    # IMPORTANTE: Usamos los coeficientes (sos_b, fir_c), NO las señales (sb, sf)
    w_iir, h_iir = sig.sosfreqz(sos_b, worN=8000, fs=4000)
    w_fir, h_fir = sig.freqz(fir_c, a=1, worN=8000, fs=4000)

    # --- Gráfico 1: Respuesta en Frecuencia ---
    plt.figure(figsize=(10, 5))
    plt.plot(w_iir, 20 * np.log10(np.abs(h_iir) + 1e-12), label='Butterworth (IIR)')
    plt.plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-12), label='FIR Rectangular')
    
    # # Plantilla
    # plt.axvline(20, color='red', linestyle='--', alpha=0.3)
    # plt.axvline(600, color='red', linestyle='--', alpha=0.3)
    # plt.axhline(-80, color='black', linestyle=':', label='Stopband -80dB')
    
    plot_plantilla(
        filter_type='bandpass',
        fpass=wp,
        ripple=gpass,
        fstop=ws,
        attenuation=gstop,
        fs=fs
    )
    
    plt.title("Comparativa de Magnitud: IIR vs FIR")
    plt.ylabel("Magnitud [dB]")
    plt.xlabel("Frecuencia [Hz]")
    plt.xlim([0, 700])
    plt.ylim([-100, 5])
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Gráfico 2: Señal en el tiempo ---
    t = np.arange(len(sb[:fs*2])) / fs

    plt.figure(figsize=(12, 6))
    plt.plot(t, original[:fs*2], label='Original', color='gray', alpha=0.3)
    plt.plot(t, sb[:fs*2], label='Butterworth (IIR)', alpha=0.8)
    plt.plot(t, sf[:fs*2], label='Ventana Rectangular (FIR)', alpha=0.8)
    plt.xlim(0.25, 0.3) # Zoom 
    plt.title(f"Comparativa de Filtros - Paciente {p_id_test}")
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()
#%%

# FUNCIONES DE ANÁLISIS ESPECTRAL (SNR-PSD)

def calcular_snr_psd(señal, fs):       # Calcula SNR usando PSD
    nperseg = min(len(señal) // 4, fs)
    if nperseg < 64:
        nperseg = 64
        
    f, Pxx = sig.welch(señal, fs, window='hamming', nperseg=nperseg)
    df = f[1] - f[0]
    
    # Energía en bandas
    mask_fund = (f >= 20) & (f <= 150)      # S1/S2
    mask_soplo = (f >= 150) & (f <= 750)     # Soplos
    
    E_fund = np.sum(Pxx[mask_fund] * df)
    E_soplo = np.sum(Pxx[mask_soplo] * df)
    
    # SNR
    snr = 10 * np.log10(E_fund / (E_soplo + 1e-12))
    
    return snr, E_fund, E_soplo, f, Pxx


# FUNCIONES DE ANÁLISIS TEMPORAL (ENVOLVENTES Y PICOS)

def calcular_envolvente_rms(señal, fs, ventana_ms=50):   # Calcula envolvente RMS de la señal
    ventana_samples = int(fs * ventana_ms / 1000)
    # Asegurar que la ventana sea impar
    if ventana_samples % 2 == 0:
        ventana_samples += 1
    
    # Calcular RMS con ventana deslizante
    window = np.ones(ventana_samples) / ventana_samples
    env_rms = np.sqrt(np.convolve(señal**2, window, mode='same'))
    
    # Normalizar
    env_rms = env_rms / (np.max(env_rms) + 1e-12)
    
    return env_rms

def detectar_picos_cardiacos(env_rms, fs, altura_min=0.05, distancia_min_ms=250):
    distancia_min = int(distancia_min_ms * fs / 1000)
    
    # Bajamos la altura a 0.05 y usamos una prominencia de 0.06
    # Esto atrapará los latidos débiles del paciente sano
    peaks, propiedades = find_peaks(env_rms, 
                                     height=altura_min, 
                                     distance=distancia_min,
                                     prominence=0.06) 
    return peaks, propiedades

def calcular_metricas_temporales(peaks, fs):    # Calcula métricas temporales a partir de los picos detectados
    if len(peaks) < 2:
        return {
            'frecuencia_cardiaca': 0,
            'intervalo_promedio': 0,
            'intervalo_std': 0,
            'num_latidos': len(peaks)
        }
    
    # Intervalos entre latidos (en segundos)
    intervalos = np.diff(peaks) / fs
    
    # Frecuencia cardíaca (latidos por minuto)
    fc_promedio = 60 / np.mean(intervalos)
    
    
    return {
        'frecuencia_cardiaca': fc_promedio,
        'intervalo_promedio': np.mean(intervalos),
        'intervalo_std': np.std(intervalos),
        'num_latidos': len(peaks)
    }


# FUNCIÓN DE PROCESAMIENTO COMPLETO (ESPECTRAL + TEMPORAL)

def procesar_paciente_completo(p_id, fs):
    """
    Procesa un paciente completo: análisis espectral y temporal
    
    Retorna:
    --------
    dict : Todas las métricas del paciente
    """
    señal_filtrada, _, _, _, _ = cargar_y_filtrar(p_id)
    if señal_filtrada is None:
        return None
    
    # Análisis espectral (SNR-PSD)
    snr, E_fund, E_soplo, f, Pxx = calcular_snr_psd(señal_filtrada, fs)
    
    # Análisis temporal (envolvente y picos)
    env_rms = calcular_envolvente_rms(señal_filtrada, fs)
    peaks, _ = detectar_picos_cardiacos(env_rms, fs)
    metricas_temp = calcular_metricas_temporales(peaks, fs)
    
    return {
        'ID': p_id,
        'SNR_dB': snr,
        'E_fundamental': E_fund,
        'E_soplo': E_soplo,
        'frecuencia_cardiaca': metricas_temp['frecuencia_cardiaca'],
        'num_latidos': metricas_temp['num_latidos'],
        'intervalo_promedio': metricas_temp['intervalo_promedio'],
        'env_rms': env_rms,
        'peaks': peaks,
        'señal_filtrada': señal_filtrada,
        'f': f,
        'Pxx': Pxx
    }

#%% PROCESAR TODAS LAS SEÑALES

print("="*60)
print("PROCESANDO SEÑALES (ESPECTRAL + TEMPORAL)")
print("="*60)

resultados = []
psd_data = {}
raw_data = {}
temp_data = {}  # Para guardar datos temporales de ejemplo

# --- PROCESAR SANOS ---
print("\n--- SANOS ---")
for p_id in IDS_SANOS:
    datos = procesar_paciente_completo(p_id, fs)
    if datos is not None:
        resultados.append({
            'ID': datos['ID'], 'Clase': 'Sano', 'SNR_dB': datos['SNR_dB'],
            'E_fundamental': datos['E_fundamental'], 'E_soplo': datos['E_soplo'],
            'frecuencia_cardiaca': datos['frecuencia_cardiaca'],
            'num_latidos': datos['num_latidos']
        })
        
        # GUARDADO PARA GRÁFICOS: Si coincide con nuestra elección
        if p_id == ID_REPRESENTATIVO_SANO:
            psd_data['sano'] = {'f': datos['f'], 'Pxx': datos['Pxx'], 'señal': datos['señal_filtrada'], 'ID': p_id, 'snr': datos['SNR_dB']}
            temp_data['sano'] = {'env_rms': datos['env_rms'], 'peaks': datos['peaks']}
            print(f" {p_id}: SNR={datos['SNR_dB']:.2f} dB -->SELECCIONADO PARA GRÁFICOS")
        else:
            print(f" {p_id}: SNR={datos['SNR_dB']:.2f} dB")

# --- PROCESAR SOPLOS ---
print("\n--- SOPLOS ---")
for p_id in IDS_SOPLO:
    datos = procesar_paciente_completo(p_id, fs)
    if datos is not None:
        resultados.append({
            'ID': datos['ID'], 'Clase': 'Soplo', 'SNR_dB': datos['SNR_dB'],
            'E_fundamental': datos['E_fundamental'], 'E_soplo': datos['E_soplo'],
            'frecuencia_cardiaca': datos['frecuencia_cardiaca'],
            'num_latidos': datos['num_latidos']
        })
        
        # GUARDADO PARA GRÁFICOS: Si coincide con nuestra elección
        if p_id == ID_REPRESENTATIVO_SOPLO:
            psd_data['soplo'] = {'f': datos['f'], 'Pxx': datos['Pxx'], 'señal': datos['señal_filtrada'], 'ID': p_id, 'snr': datos['SNR_dB']}
            temp_data['soplo'] = {'env_rms': datos['env_rms'], 'peaks': datos['peaks']}
            print(f" {p_id}: SNR={datos['SNR_dB']:.2f} dB -->SELECCIONADO PARA GRÁFICOS")
        else:
            print(f" {p_id}: SNR={datos['SNR_dB']:.2f} dB")
# Crear DataFrame
df = pd.DataFrame(resultados)

print(f"\n✓ Procesados {len(df)} pacientes")
print(f"  Sanos: {len(df[df['Clase'] == 'Sano'])}")
print(f"  Soplos: {len(df[df['Clase'] == 'Soplo'])}")

# PREPARAR DATOS PARA GRÁFICOS

data_sanos = df[df['Clase'] == 'Sano']['SNR_dB'].values
data_soplos = df[df['Clase'] == 'Soplo']['SNR_dB'].values

# Estadísticas SNR
media_sanos = np.mean(data_sanos)
std_sanos = np.std(data_sanos)
mediana_sanos = np.median(data_sanos)

media_soplos = np.mean(data_soplos)
std_soplos = np.std(data_soplos)
mediana_soplos = np.median(data_soplos)

# Umbral óptimo
umbral = (mediana_sanos + mediana_soplos) / 2

# Clasificación por SNR
df['Prediccion_SNR'] = df['SNR_dB'].apply(lambda x: 'Soplo' if x > umbral else 'Sano')

# Matriz de confusión (SNR)
TP_snr = len(df[(df['Clase'] == 'Soplo') & (df['Prediccion_SNR'] == 'Soplo')])
TN_snr = len(df[(df['Clase'] == 'Sano') & (df['Prediccion_SNR'] == 'Sano')])
FP_snr = len(df[(df['Clase'] == 'Sano') & (df['Prediccion_SNR'] == 'Soplo')])
FN_snr = len(df[(df['Clase'] == 'Soplo') & (df['Prediccion_SNR'] == 'Sano')])

sens_snr = TP_snr / (TP_snr + FN_snr) if (TP_snr + FN_snr) > 0 else 0
spec_snr = TN_snr / (TN_snr + FP_snr) if (TN_snr + FP_snr) > 0 else 0
prec_snr = TP_snr / (TP_snr + FP_snr) if (TP_snr + FP_snr) > 0 else 0
f1_snr = 2 * (prec_snr * sens_snr) / (prec_snr + sens_snr) if (prec_snr + sens_snr) > 0 else 0

#%% GRÁFICO: SEÑAL ORIGINAL VS FILTRADA


señal_sano_raw = cargar_señal_raw(PACIENTE_SANO)
señal_sano_filtrada, _, _, _, _ = cargar_y_filtrar(PACIENTE_SANO)
señal_soplo_raw = cargar_señal_raw(PACIENTE_SOPLO)
señal_soplo_filtrada, _, _, _, _ = cargar_y_filtrar(PACIENTE_SOPLO)

if señal_sano_filtrada is not None:
    snr_sano, _, _, _, _ = calcular_snr_psd(señal_sano_filtrada, fs)
else:
    snr_sano = 0

if señal_soplo_filtrada is not None:
    snr_soplo, _, _, _, _ = calcular_snr_psd(señal_soplo_filtrada, fs)
else:
    snr_soplo = 0

plt.figure(figsize=(14, 10))

# Paciente Sano - Vista completa
plt.subplot(2, 2, 1)
t = np.arange(len(señal_sano_raw[:fs*2])) / fs
plt.plot(t, señal_sano_raw[:fs*2], 'b-', alpha=0.5, linewidth=0.8, label='Original')
plt.plot(t, señal_sano_filtrada[:fs*2], 'g-', alpha=1, linewidth=1.2, label='Filtrada')
plt.xlabel('Tiempo [s]', fontsize=10)
plt.ylabel('Amplitud', fontsize=10)
plt.title(f'Sano ({PACIENTE_SANO}) - Original vs Filtrada\nSNR = {snr_sano:.2f} dB', fontsize=11, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(0, 2)
plt.ylim(-1,1)

# Paciente Sano - Zoom
plt.subplot(2, 2, 3)
t_zoom = np.arange(len(señal_sano_raw[:fs*1])) / fs
plt.plot(t_zoom, señal_sano_raw[:fs*1], 'b-', alpha=0.5, linewidth=0.8, label='Original')
plt.plot(t_zoom, señal_sano_filtrada[:fs*1], 'g-', alpha=1, linewidth=1.2, label='Filtrada')
plt.xlabel('Tiempo [s]', fontsize=10)
plt.ylabel('Amplitud', fontsize=10)
plt.title('Zoom - Sano', fontsize=11, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(0.2, 0.35)
plt.ylim(-1,1)

# Paciente Soplo - Vista completa
plt.subplot(2, 2, 2)
t = np.arange(len(señal_soplo_raw[:fs*2])) / fs
plt.plot(t, señal_soplo_raw[:fs*2], 'b-', alpha=0.5, linewidth=0.8, label='Original')
plt.plot(t, señal_soplo_filtrada[:fs*2], 'r-', alpha=1, linewidth=1.2, label='Filtrada')
plt.xlabel('Tiempo [s]', fontsize=10)
plt.ylabel('Amplitud', fontsize=10)
plt.title(f'Soplo ({PACIENTE_SOPLO}) - Original vs Filtrada\nSNR = {snr_soplo:.2f} dB', fontsize=11, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(0, 2)
plt.ylim(-1,1)

# Paciente Soplo - Zoom
plt.subplot(2, 2, 4)
t_zoom = np.arange(len(señal_soplo_raw[:fs*1])) / fs
plt.plot(t_zoom, señal_soplo_raw[:fs*1], 'b-', alpha=0.5, linewidth=0.8, label='Original')
plt.plot(t_zoom, señal_soplo_filtrada[:fs*1], 'r-', alpha=1, linewidth=1.2, label='Filtrada')
plt.xlabel('Tiempo [s]', fontsize=10)
plt.ylabel('Amplitud', fontsize=10)
plt.title('Zoom - Soplo', fontsize=11, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.15)
plt.ylim(-1,1)

plt.suptitle('Efecto del Filtrado: Señal Original vs Filtrada', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#%% GRÁFICO: DETECCIÓN DE LATIDOS

if 'sano' in temp_data and 'soplo' in temp_data:
    fig, axes = plt.subplots(2, 1, figsize=(14, 12)) # Usamos subplots para mayor control
    
    # --- Paciente Sano ---
    ax1 = axes[0]
    p_sano = PACIENTE_SANO
    t_sano = np.arange(len(psd_data['sano']['señal'][:fs*5])) / fs
    env_sano = temp_data['sano']['env_rms'][:fs*5]
    peaks_sano = temp_data['sano']['peaks']
    peaks_sano_plot = peaks_sano[peaks_sano < fs*5]
    
    ax1.plot(t_sano, psd_data['sano']['señal'][:fs*5], 'gray', alpha=0.3, label='Señal')
    ax1.plot(t_sano, env_sano, 'b-', linewidth=2, label='Envolvente RMS')
    ax1.scatter(peaks_sano_plot/fs, env_sano[peaks_sano_plot], color='red', s=50, zorder=5, label=f'Picos S1/S2 ({len(peaks_sano_plot)})')
    ax1.set_title(f'Paciente Sano ({PACIENTE_SANO}) - Detección de Latidos Cardíacos\nFrecuencia Cardíaca: {df[df["ID"]==PACIENTE_SANO]["frecuencia_cardiaca"].values[0]:.0f} lpm', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Tiempo [s]', fontsize=14)
    ax1.set_ylabel('Amplitud', fontsize=14)
    ax1.set_ylim(-1,1)
    ax1.set_xlim(0,5)
    ax1.grid(True)
    ax1.legend()

    # --- Paciente Soplo ---
    ax2 = axes[1]
    p_soplo = PACIENTE_SOPLO
    t_soplo = np.arange(len(psd_data['soplo']['señal'][:fs*5])) / fs
    env_soplo = temp_data['soplo']['env_rms'][:fs*5]
    peaks_soplo = temp_data['soplo']['peaks']
    peaks_soplo_plot = peaks_soplo[peaks_soplo < fs*5]
    
    ax2.plot(t_soplo, psd_data['soplo']['señal'][:fs*5], 'gray', alpha=0.3, label='Señal')
    ax2.plot(t_soplo, env_soplo, 'r-', linewidth=2, label='Envolvente RMS')
    ax2.scatter(peaks_soplo_plot/fs, env_soplo[peaks_soplo_plot], color='blue', s=50, zorder=5, label=f'Picos S1/S2 ({len(peaks_soplo_plot)})')
    ax2.set_title(f'Paciente con Soplo ({PACIENTE_SOPLO}) - Detección de Latidos Cardíacos\nFrecuencia Cardíaca: {df[df["ID"]==PACIENTE_SOPLO]["frecuencia_cardiaca"].values[0]:.0f} lpm', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Tiempo [s]', fontsize=14)
    ax2.set_ylabel('Amplitud', fontsize=14)
    ax2.set_ylim(-1,1)
    ax2.set_xlim(0,5)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show() 
else:
    print("ERROR: No se graficó porque faltan llaves en temp_data.")
#%% GRÁFICO: BOXPLOT DE SNR

plt.figure(figsize=(10, 6))

bp = plt.boxplot([data_sanos, data_soplos], 
                  labels=['Sanos (n=10)', 'Soplos (n=11)'],
                  patch_artist=True, widths=0.6)

bp['boxes'][0].set_facecolor('#90EE90')
bp['boxes'][1].set_facecolor('#FAA0A0')
bp['medians'][0].set_color('darkgreen')
bp['medians'][1].set_color('darkred')
bp['medians'][0].set_linewidth(2)
bp['medians'][1].set_linewidth(2)

plt.scatter(np.ones(len(data_sanos)) * 1, data_sanos, 
            color='green', alpha=0.6, s=60, zorder=3, edgecolors='black', linewidth=1)
plt.scatter(np.ones(len(data_soplos)) * 2, data_soplos, 
            color='red', alpha=0.6, s=60, zorder=3, edgecolors='black', linewidth=1)

plt.axhline(y=umbral, color='black', linestyle='--', linewidth=2, 
            label=f'Umbral: {umbral:.2f} dB')

plt.ylabel('SNR [dB]', fontsize=12)
plt.title('Distribución de SNR-PSD: Sanos vs Soplos', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

#%% GRÁFICO: HISTOGRAMA
plt.figure(figsize=(10, 6))

bins = np.linspace(-16, 12, 20)

plt.subplot(2,1,1)
plt.hist(data_sanos, bins=bins, alpha=0.6, label='Sanos', 
         color='green', edgecolor='black', linewidth=1)

plt.axvline(x=umbral, color='black', linestyle='--', linewidth=2, 
            label=f'Umbral: {umbral:.2f} dB')

plt.xlabel('SNR [dB]', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.title('Histograma de SNR-PSD por Clase', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2,1,2)
plt.hist(data_soplos, bins=bins, alpha=0.6, label='Soplos', 
         color='red', edgecolor='black', linewidth=1)

plt.axvline(x=umbral, color='black', linestyle='--', linewidth=2, 
            label=f'Umbral: {umbral:.2f} dB')

plt.xlabel('SNR [dB]', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% GRÁFICO: ESPECTROS COMPARATIVOS

if 'sano' in psd_data and 'soplo' in psd_data:
    plt.figure(figsize=(10, 6))
    
    # Espectros
    psd_sano_norm = psd_data['sano']['Pxx'] / np.max(psd_data['sano']['Pxx'])
    psd_soplo_norm = psd_data['soplo']['Pxx'] / np.max(psd_data['soplo']['Pxx'])
    
    plt.semilogy(psd_data['sano']['f'], psd_sano_norm, 'g-', linewidth=1.5, 
                 label=f'Sano ({psd_data["sano"]["ID"]}, SNR={psd_data["sano"]["snr"]:.1f}dB)')
    plt.semilogy(psd_data['soplo']['f'], psd_soplo_norm, 'r-', linewidth=1.5,
                 label=f'Soplo ({psd_data["soplo"]["ID"]}, SNR={psd_data["soplo"]["snr"]:.1f}dB)')
    plt.axvspan(20, 150, alpha=0.2, color='green', label='S1/S2 (20-150 Hz)')
    plt.axvspan(150, 600, alpha=0.2, color='red', label='Soplo (150-600 Hz)')
    plt.xlim(0, 800)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD Normalizada')
    plt.title('Espectros Comparativos')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#%% GRÁFICO: FRECUENCIA CARDÍACA vs SNR

plt.figure(figsize=(10, 6))

for clase in ['Sano', 'Soplo']:
    mask = df['Clase'] == clase
    color = 'green' if clase == 'Sano' else 'red'
    plt.scatter(df[mask]['frecuencia_cardiaca'], df[mask]['SNR_dB'], 
                c=color, label=clase, s=80, alpha=0.6, edgecolors='black', linewidth=1)

plt.axhline(y=umbral, color='black', linestyle='--', linewidth=1.5, label=f'Umbral SNR: {umbral:.2f} dB')
plt.xlabel('Frecuencia Cardíaca [lpm]', fontsize=12)
plt.ylabel('SNR [dB]', fontsize=12)
plt.title('Relación entre Frecuencia Cardíaca y SNR-PSD', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% IMPRIMIR ESTADÍSTICAS COMPLETAS

print("\n" + "="*70)
print("RESULTADOS DEL ANÁLISIS COMPLETO")
print("="*70)

print("\nESTADÍSTICAS DE SNR-PSD POR CLASE:")
print("-" * 50)
print(f"{'Clase':<10} {'Cant':<8} {'Media':<10} {'Std':<10} {'Mediana':<10} {'Min':<10} {'Max':<10}")
print("-" * 50)
print(f"{'Sanos':<10} {len(data_sanos):<8} {media_sanos:<10.2f} {std_sanos:<10.2f} {mediana_sanos:<10.2f} {np.min(data_sanos):<10.2f} {np.max(data_sanos):<10.2f}")
print(f"{'Soplos':<10} {len(data_soplos):<8} {media_soplos:<10.2f} {std_soplos:<10.2f} {mediana_soplos:<10.2f} {np.min(data_soplos):<10.2f} {np.max(data_soplos):<10.2f}")
print("-" * 50)

print("\nESTADÍSTICAS TEMPORALES (Frecuencia Cardíaca):")
print("-" * 50)
fc_sanos = df[df['Clase'] == 'Sano']['frecuencia_cardiaca'].values
fc_soplos = df[df['Clase'] == 'Soplo']['frecuencia_cardiaca'].values
print(f"{'Sanos':<10} Media: {np.mean(fc_sanos):.1f} lpm | Std: {np.std(fc_sanos):.1f} | Min: {np.min(fc_sanos):.1f} | Max: {np.max(fc_sanos):.1f}")
print(f"{'Soplos':<10} Media: {np.mean(fc_soplos):.1f} lpm | Std: {np.std(fc_soplos):.1f} | Min: {np.min(fc_soplos):.1f} | Max: {np.max(fc_soplos):.1f}")
print("-" * 50)

print("\nCLASIFICACIÓN POR SNR (Umbral = {:.2f} dB):".format(umbral))
print("-" * 50)
print(f"Sensibilidad:   {sens_snr:.2%} ({TP_snr}/{TP_snr+FN_snr})")
print(f"Especificidad: {spec_snr:.2%} ({TN_snr}/{TN_snr+FP_snr})")
print(f"Precisión:      {prec_snr:.2%} ({TP_snr}/{TP_snr+FP_snr})")
print(f"F1-Score:       {f1_snr:.2%}")

print("\nMATRIZ DE CONFUSIÓN (SNR):")
print("-" * 30)
print(f"{'':<12} {'Predicción':>12}")
print(f"{'':<12} {'Sano':>6} {'Soplo':>6}")
print("-" * 30)
print(f"{'Real Sano':<12} {TN_snr:>6} {FP_snr:>6}")
print(f"{'Real Soplo':<12} {FN_snr:>6} {TP_snr:>6}")
print("-" * 30)

print("\nPACIENTES BIEN CLASIFICADOS POR SNR:")
bien = df[df['Clase'] == df['Prediccion_SNR']]
for _, row in bien.iterrows():
    print(f"  {row['ID']} ({row['Clase']}): SNR={row['SNR_dB']:.2f} dB | FC={row['frecuencia_cardiaca']:.0f} lpm")

print("\nPACIENTES MAL CLASIFICADOS POR SNR:")
mal = df[df['Clase'] != df['Prediccion_SNR']]
for _, row in mal.iterrows():
    print(f"  {row['ID']} (Real: {row['Clase']} → Predicho: {row['Prediccion_SNR']}): SNR={row['SNR_dB']:.2f} dB | FC={row['frecuencia_cardiaca']:.0f} lpm")

