import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class actividad_2:
    def __init__(self):
        self.ruta_resultados = "actividad_2/resultados_actividad_2/"
        self.ruta_imagenes = os.path.join(self.ruta_resultados, "Imagenes_puntos_11_21_PNG/")
        # Crear las carpetas 
        os.makedirs(self.ruta_resultados, exist_ok=True)
        os.makedirs(self.ruta_imagenes, exist_ok=True)
actividad=actividad_2()

df = pd.DataFrame(columns=["# ejercicio", "resultado"])

def generar_resultado(ejercicio, resultado):
    global df
    df.loc[len(df)] = [ejercicio, resultado]

# Punto 1
array1 = np.arange(10, 30)
generar_resultado(1, str(array1))

# Punto 2
matriz_ones = np.ones((10, 10))
suma = np.sum(matriz_ones)
generar_resultado(2, suma)

# Punto 3
array2 = np.random.randint(1, 11, 5)
array3 = np.random.randint(1, 11, 5)
producto = array2 * array3
generar_resultado(3, str(producto))

# Punto 4
matriz_ij = np.random.randint(1, 10, (4, 4))
determinante = np.linalg.det(matriz_ij)
if determinante != 0:
    inversa = np.linalg.inv(matriz_ij)
    generar_resultado(4, f"Inversa: {inversa}")
else:
    generar_resultado(4, "La matriz no es invertible")

# Punto 5
array_random = np.random.random(100)
max_index = np.argmax(array_random)
min_index = np.argmin(array_random)
generar_resultado(5, f"Máximo: {array_random[max_index]}, Índice: {max_index}")
generar_resultado(5, f"Mínimo: {array_random[min_index]}, Índice: {min_index}")

# Punto 6
array_a = np.arange(3).reshape(3, 1)
array_b = np.arange(3).reshape(1, 3)
resultado_broadcast = array_a + array_b
generar_resultado(6, str(resultado_broadcast))

# Punto 7
matriz_5x5 = np.random.randint(1, 10, (5, 5))
submatriz = matriz_5x5[1:3, 1:3]
generar_resultado(7, str(submatriz))

# Punto 8
array_ceros = np.zeros(10)
array_ceros[3:7] = 5
generar_resultado(8, str(array_ceros))

# Punto 9
matriz_3x3 = np.random.randint(1, 10, (3, 3))
matriz_inversa = matriz_3x3[::-1]
generar_resultado(9, str(matriz_inversa))

# Punto 10
datos_random = np.random.random(10)
mayores_05 = datos_random[datos_random > 0.5]
generar_resultado(10, str(mayores_05))

# Guardar el archivo CSV en la carpeta correcta
actividad = actividad_2()
csv_path = os.path.join(actividad.ruta_resultados, "resultados_actividad_2.csv")
df.to_csv(csv_path, index=False)


# Generar y guardar imágenes de los graficos en las carpetas indicadas
def guardar_grafico(nombre):
    ruta_completa = os.path.join(actividad.ruta_imagenes, nombre)
    plt.savefig(ruta_completa)
    plt.close()
   
# Punto 11
x = np.random.random(100)
y = np.random.random(100)
plt.scatter(x, y)
# Agregar etiquetas y título
plt.xlabel('Valores de X')
plt.ylabel('Valores de Y')
plt.title('Gráfico de Dispersión de Números Aleatorios')
plt.grid()
guardar_grafico("punto_11.png")

# Punto 12
x_vals = np.linspace(-2*np.pi, 2*np.pi, 100)
y_vals = np.sin(x_vals) + np.random.normal(0, 0.1, 100)
plt.scatter(x_vals, y_vals, label='y = sin(x) + ruido')
plt.plot(x_vals, np.sin(x_vals), color='red', label='y = sin(x)')
# Agregar etiquetas y leyenda
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfico de Dispersión con Ruido Gaussiano')
plt.legend()
guardar_grafico("punto_12.png")

# Punto 13
x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
z_grid = np.cos(x_grid) + np.sin(y_grid)
plt.contour(x_grid, y_grid, z_grid)
# Agregar etiquetas y título
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de Contorno de z = cos(x) + sin(y)')
guardar_grafico("punto_13.png")

# Punto 14
x_d = np.random.randn(1000)
y_d = np.random.randn(1000)
plt.scatter(x_d, y_d, c=np.sqrt(x_d**2 + y_d**2), cmap='plasma')
# Agregar barra de colores y etiquetas
plt.colorbar(label='Densidad')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de Dispersión con Densidad de Puntos')
guardar_grafico("punto_14.png")

# Punto 15
plt.contourf(x_grid, y_grid, z_grid, cmap='plasma')
# Agregar barra de colores y etiquetas
plt.colorbar(label='Valor de z')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de Contorno Lleno de $z = cos(x) + sin(y)$')
guardar_grafico("punto_15.png")

# Punto 16
plt.scatter(x_vals, y_vals, label=r'$y = \sin(x) + ruido$', alpha=0.6)
plt.plot(x_vals, np.sin(x_vals), color='red', label=r'$y = \sin(x)$')
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.title("Gráfico de Dispersión")
plt.legend()
guardar_grafico("punto_16.png")

# Punto 17
data_hist = np.random.randn(1000)
plt.hist(data_hist, bins=30, alpha=0.7)
# Etiquetas y título
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma de una Distribución Normal')
guardar_grafico("punto_17.png")

# Punto 18
# Generar 1000 números aleatorios para cada conjunto con distribuciones normales diferentes
data1 = np.random.normal(loc=0, scale=1, size=1000)   # Media = 0, Desviación estándar = 1
data2 = np.random.normal(loc=3, scale=1.5, size=1000) # Media = 3, Desviación estándar = 1.5
# Crear el histograma
plt.figure(figsize=(8, 6))
plt.hist(data1, bins=30, color='skyblue', edgecolor='black', alpha=0.6, label='Media = 0')
plt.hist(data2, bins=30, color='salmon', edgecolor='black', alpha=0.6, label='Media = 0')
# Etiquetas y título
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma de dos distribuciones normales')
plt.legend()
guardar_grafico("punto_18.png")

# Punto 19
# Generar datos con distribución normal
data = np.random.normal(loc=0, scale=1, size=1000)  # Media = 0, Std = 1

# Definir diferentes valores de bins
bins_values = [10, 30, 50]

# Crear subgráficos para comparar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, bins in enumerate(bins_values):
    axes[i].hist(data, bins=bins, color='salmon', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Histograma con {bins} bins')
    axes[i].set_xlabel('Valor')
    axes[i].set_ylabel('Frecuencia')
guardar_grafico("punto_19.png")

# Punto 20
# Generar datos con distribución normal
data = np.random.normal(loc=0, scale=1, size=1000)  # Media = 0, Desviación estándar = 1

# Calcular la media de los datos
mean_value = np.mean(data)

# Definir diferentes valores de bins
bins_values = [10, 30, 50]

# Crear subgráficos para comparar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, bins in enumerate(bins_values):
    axes[i].hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    axes[i].axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Media: {mean_value:.2f}')
    axes[i].set_title(f'Histograma con {bins} bins')
    axes[i].set_xlabel('Valor')
    axes[i].set_ylabel('Frecuencia')
    axes[i].legend()
plt.legend()
guardar_grafico("punto_20.png")

# Punto 21
# Generar dos sets de datos con distribuciones normales diferentes
data1 = np.random.normal(loc=0, scale=1, size=1000)   # Media = 0, Desviación estándar = 1
data2 = np.random.normal(loc=3, scale=1.5, size=1000) # Media = 3, Desviación estándar = 1.5

# Crear el histograma superpuesto
plt.figure(figsize=(10, 6))
plt.hist(data1, bins=30, color='blue', alpha=0.5, label='Distribución 1 (Media=0, STD=1)')
plt.hist(data2, bins=30, color='orange', alpha=0.5, label='Distribución 2 (Media=3, STD=1.5)')

# Etiquetas y leyenda
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histogramas Superpuestos de Dos Distribuciones Normales')
plt.legend()
plt.grid(True)

plt.legend()
guardar_grafico("punto_21.png")