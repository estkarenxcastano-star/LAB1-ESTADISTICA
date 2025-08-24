# LABORATORIO 1- ANALISIS ESTADISTICO DE LA SEÑAL
## Resumen
En el presente informe se quiere analizar estadísticamente una señal biomedica utilizando herramiemtas de programacion de Python. Para ello se busca identificar,calcular y representar estadísticos descriptivos que permiten caracterizar una señal.

# PARTE A
En esta primera parte se debia descargar una señal de Physionet, importarla en Phython para graficarla y a partir de esto calcular sus estadísticos descriptivos, lo que son la media, desviación estándar, coeficiente de variación, histogramas, funcion de probabilidad y curtosis. 
## Se realizo el siguiente algoritmo
![WhatsApp Image 2025-08-24 at 09 26 13](https://github.com/user-attachments/assets/b2a0280e-e4b1-4c46-8c10-950cb7d5ae2a)

## LIBRERIAS
Las librerias que implementamos fueron las siguientes:

+ **Importación de librerias**
```phyton
import numpy as np
import matplotlib.pyplot as plt
!pip install wfdb
from scipy.stats import kurtosis
import wfdb
```
### A partir de esto importamos la señal obtenida de Physionet
+ **Importación señal a Google Colab**
```phyton
record = wfdb.rdrecord('mitdb/100', pn_dir='mitdb', sampto=1000)
signal = record.p_signal[:, 0]
plt.figure(figsize=(10, 4))
plt.plot(signal)
plt.title('Señal fisiológica (ECG)')
plt.xlabel('Tiempo (muestras)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()
```  



