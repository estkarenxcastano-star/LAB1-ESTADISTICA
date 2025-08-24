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
+ **Media:  -0.31188**
+ **Desviación estándar:  0.18336**
+ **Coeficiente de variación:  58.78942**
+ **Curtosis:  24.33657**
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

### Ahora calculamos los estadísticos descriptivos con funciones predefinidas de Python
+ **Media estándar**
```python
media_np = np.mean(signal)
media_np
```
+ **Desviación estándar**
```python
desv_std_np = np.std (signal, ddof=1)
desv_std_np
```
+ **Coeficiente de variación**
```python
cv_np = (desv_std_np / media_np) * 100
cv_np
```
+ **Curtosis**
```python
curtosis_scipy = kurtosis(signal, fisher=True, bias=False)
curtosis_scipy
```
### Obteniendo como resultados
+ **Media:  -0.311885**
+ **Desviación estándar:  0.18335536797695878**
+ **Coeficiente de variación:  -58.789415321980464**
+ **Curtosis:  24.464720388248345**
### Histograma
```python
min_val = np.min(signal)
max_val = np.max(signal)
plt.figure(figsize=(10,4))
plt.hist(signal, bins=100, range=(min_val, max_val), edgecolor="black", alpha=0.7)
plt.xlabel("Amplitud")
plt.ylabel("Frecuencia")
plt.title("Histograma")
plt.grid(axis="y", alpha=0.5)
plt.show()
```
### Obteniendo la siguiente gráfica
<img width="847" height="391" alt="image" src="https://github.com/user-attachments/assets/96d3cf78-c957-44ad-90c1-3872aa5e6d2c" />

# PARTE B
Se generó una señal, con el generador de señales fisiológicas similar a la de la Parte A y esta se capturó mediante un DAQ con el driver NI MAX. Luego, la señal fue importada en Python, graficada, se calcularon sus estadísticos descriptivos y se compararon con los resultados de la Parte A.
## Se realizo el siguiente algoritmo
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/8c0b3cf9-3305-4fe0-a9b7-f1607f008cb8" />

+ **El código que se utilizó para extraer la señal del DAQ fue el siguiente**
```python
!pip install nidaqmx
import nidaqmx
import numpy as np
import pandas as pd
canal = "Dev1/ai0"       
fs = 10000               
num_muestras = 5000     
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(canal) 
    task.timing.cfg_samp_clk_timing(fs, samps_per_chan=num_muestras)
    data = task.read(number_of_samples_per_channel=num_muestras)
    data = np.array(data)
tiempo = np.linspace(0, num_muestras/fs, num_muestras, endpoint=False)
df = pd.DataFrame({"Tiempo (s)": tiempo, "Voltaje (V)": data})
df.to_csv("senal_DAQ.csv", index=False)
print("✅ Señal guardada en 'senal_DAQ.csv'")
df.to_csv("senal_DAQ.txt", sep="\t", index=False)
df.to_feather("senal_DAQ.feather")
```

+ **Los códigos utilizados para leer la señal desde el archivo que se extrajo y ver los datos son los siguientes:**
```python
import numpy as np

señal = np.loadtxt("senal_DAQ.txt", skiprows=1)
print(señal[:10])
```
```python
import pandas as pd

señal = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/senal_DAQ.txt", sep = "\s+", header=None, skiprows=1)

print(señal.head())
```
+ **Ahora se grafica la señal que obtuvimos del generador de señales fisiológicas**
```python
import pandas as pd

señal = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/senal_DAQ.txt", sep = "\s+", header=None, skiprows=1)

print(señal.head())
```
### Para obtener la siguiente gráfica
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/a2856286-ffea-49b9-83e1-1d75f89d8a6f" />

### Ahora calcularemos los datos estádisticos de la señal
```python
import pandas as pd
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/senal_DAQ.txt",
                 sep=r"\s+", header=None, skiprows=1, names=["Tiempo", "Voltaje"])
print(df.head())

senal = df["Voltaje"].astype(float).to_numpy()
print(type(senal), senal.dtype)
print("Primeros valores:", senal[:10])


# a. Media
media_2 = np.mean(senal)

# b. Desviación estándar
desv = np.std(senal)

# c. Coeficiente de variación
CV = desv / media_2 if media_2 != 0 else np.nan


print("Datos estadísticos")
print(f"Media: {media_2}")
print(f"Desviación estándar: {desv}")
print(f"Coeficiente de variación: {CV}")
```
+ **Obtenemos los siguientes resultados**
+ **Media:  -0.1948625739533105**
+ **Desviación estándar:  0.3585715646805795**
+ **Coeficiente de variación:  -1.8401253632546906**
### Histograma
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/senal_DAQ.txt", sep=r"\s+", header=None, skiprows=1, names=["Tiempo", "Voltaje"])

print(df.head(10))
print(df.info())

senal = df["Voltaje"].astype(float).to_numpy()
print("Primeros valores de amplitud", senal[:10])
# Histograma

min_val = np.min(senal)
max_val = np.max(senal)

plt.figure(figsize=(10,5))
plt.hist(senal, bins=100, range=(min_val, max_val), edgecolor="black")
plt.xlabel("Amplitud [V]")
plt.ylabel("Frecuencia")
plt.title("Histograma de la señal generada")
plt.grid(True)
plt.show()
```
+ **Obtenemos la siguiente gráfica**
<img width="858" height="468" alt="image" src="https://github.com/user-attachments/assets/f1639a8e-5ef0-4b53-bc2c-76310ab25100" />

## Comparación de Datos Estadísticos
| **Métrica**              | **Señal A** | **Señal B** | **Interpretación** |
|-------------------------|----------------|----------------|---------------------|
| **Media**               | -0.3119        | -0.1949        | Ambas negativas; la señal B está más cercana a cero. |
| **Desv. Estándar**      | 0.1834         | 0.3586         | La señal B tiene el doble de dispersión → mayor variabilidad. |
| **Coef. Variación**    | 58.79          | -1.84          | En la señal A es muy alto, porque los picos (QRS) son muy marcados respecto a la media. En la señal B el valor es bajo y poco representativo, ya que la media cercana a cero hace que el cálculo pierda significado. |
| **Curtosis**            | 24.34          | mucho menor, según histograma        | En la señal A, la curtosis extremadamente alta indica que los valores están concentrados y aparecen picos muy definidos (QRS). En la señal B la distribución es más uniforme, sin picos extremos. |
| **Histograma**          | distribución centrada, con picos muy definidos    | Valores dispersos, amplitudes variadas, sin picos claros  |   La señal A refleja la estructura fisiológica de un ECG real; la señal B es más uniforme y ruidosa, con menos fidelidad a la realidad.  |
# PARTE C
Se debía trabajar con la señal obtenida en la PARTE B, contaminarla con distintos tipos de ruido, calcular la relación señal-ruido (SNR) en cada caso.











