# LABORATORIO 1- ANALISIS ESTADISTICO DE LA SEÑAL
## Resumen
En el presente informe se quiere analizar estadísticamente una señal biomedica utilizando herramiemtas de programacion de Python. Para ello se busca identificar,calcular y representar estadísticos descriptivos que permiten caracterizar una señal.

# PARTE A
En esta primera parte se debia descargar una señal de Physionet, importarla en Phython para graficarla y a partir de esto calcular sus estadísticos descriptivos, lo que son la media, desviación estándar, coeficiente de variación, histogramas y curtosis. 
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
### Así obtuvimos la siguiente gráfica:
+ **Gráfica señal de Physionet**
<img width="857" height="391" alt="image" src="https://github.com/user-attachments/assets/ca42a810-f6c0-42ea-b18a-65417c46bf14" />

### Luego calculamos los estadísticos descriptivos desde cero
+ **Media de la señal manual**
```phyton
suma = 0
contador = 0
for valor in signal:
  suma += valor
  contador += 1
  print (f"Sumando valor {contador}: {valor:.5f} -> suma parcial = {suma:.5}")
media = suma / contador
print (f"Media = {media:.5f}")
``` 
+ **Desviación estándar de la señal manual**
```phyton
suma_cuadrados = 0
for valor in signal:
  diferencia = valor - media
  suma_cuadrados += diferencia ** 2
  print (f"valor = {valor:.5f}, diferencia = {diferencia:.5f}, suma_cuadrados:.5f")
desv_mustral = (suma_cuadrados / (contador - 1)) ** 0.5
print (f"Desviación estándar muestral = {desv_mustral:.5f}")
```
+ **Coeficiente de variación manual**
```phyton
coef_var = (desv_mustral/abs(media)) * 100
print (f"Coeficiente de variación = {coef_var:.5f}")
```
+ **Curtosis manual**
```python
suma_m2 = 0
suma_m4 = 0
for valor in signal:
  dif = valor - media
  suma_m2 += dif ** 2
  suma_m4 += dif ** 4
  print (f"valor = {valor:.5f}, m2 parcial = {suma_m2:.5f}, m4 parcial = {suma_m4:.5f}")
  m2 = suma_m2 / contador
  m4 = suma_m4 / contador
  curtosis = (m4 / (m2 ** 2)) - 3
  print (f"Curtosis = {curtosis:.5f}")
```
### Obteniendo como resultados:
+ **Media:-0.31188**
+ **Desviación estándar:0.18336**
+ **Coeficiente de variación:58.78942**
+ **Curtosis:24.33657**
### Histograma manual
```python
def calcular_barras(datos):
  datos_ordenados = sorted(datos)
  n = 0
  for _ in datos_ordenados:
    n += 1

  indice_q1 = int(0.25 * n)
  indice_q3 = int(0.75 *n)
  q1 = datos_ordenados [indice_q1]
  q3 = datos_ordenados [indice_q3]
  iqr = q3 - q1

  ancho_barra = 2 * iqr / (n ** (1/3))
  if ancho_barra == 0:
    ancho_barra = (max(datos_ordenados) - min(datos_ordenados)) / (n ** 0.5)

  num_barras = int ((max(datos_ordenados) - min(datos_ordenados)) / ancho_barra)
  return max (1, num_barras)

num_barras = calcular_barras(signal)

valor_min = min(signal)
valor_max = max(signal)
ancho_barra = (valor_max - valor_min) / num_barras
frecuencias = [0] * num_barras
for valor in signal:
  indice = int ((valor - valor_min) / ancho_barra)
  if indice == num_barras:
    indice -= 1
  frecuencias [indice] += 1

centros_barras = []
contador_barras = 0
while contador_barras < num_barras:
  centro = valor_min + (contador_barras + 0.5) * ancho_barra
  centros_barras.append(centro)
  contador_barras += 1

plt.bar(centros_barras, frecuencias, width=ancho_barra, edgecolor="black", alpha=0.7)
plt.xlabel("Amplitud")
plt.ylabel("Frecuencia")
plt.title("Histograma manual")
plt.grid(axis="y", alpha=0.5)
plt.show
```
### Obteniendo la siguiente gráfica
<img width="569" height="450" alt="image" src="https://github.com/user-attachments/assets/0aa52329-41a9-44f0-8831-7f158e75e188" />








