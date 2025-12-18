# Simuaci-n_Amortiguador_Cinta_Industrial-
Simulación en Python de un amortiguador viscoelástico para cintas transportadoras. Modela el sistema masa-resorte ante paradas de emergencia, resolviendo la dinámica del impacto, disipación de energía y optimización de parámetros (k, c) para minimizar roturas en líneas de embotellado.

# Simulación de Amortiguador Industrial para Cintas Transportadoras 🍾⚙️

<img width="1472" height="928" alt="image" src="https://github.com/user-attachments/assets/28012f93-58a7-481b-bd5c-9e4e6bce2c01" />
<img width="3000" height="1800" alt="figura_1_respuesta_dinamica" src="https://github.com/user-attachments/assets/8db45419-fffd-42b4-9b3d-a0877849f765" />
<img width="3000" height="1800" alt="figura_2_balance_energia" src="https://github.com/user-attachments/assets/aa4f5c11-2a3b-4d0e-998f-4de34dcf965f" />
<img width="2700" height="2100" alt="figura_3_heatmap_optimizacion" src="https://github.com/user-attachments/assets/66af5460-e663-4946-af26-24d2a889fdd3" />


Herramienta computacional para modelar, simular y optimizar el comportamiento de **topes amortiguadores viscoelásticos** utilizados en líneas de envasado industrial. El proyecto analiza la respuesta dinámica de un tren de botellas ante una parada de emergencia de la cinta transportadora.

## 📋 Descripción del Proyecto

Este software aborda el problema de las paradas repentinas en líneas de producción, donde la inercia de las botellas genera impactos que pueden causar roturas.  
Se modela el sistema como un conjunto **masa-resorte-amortiguador** de un grado de libertad, permitiendo predecir el desplazamiento máximo, el tiempo de asentamiento y la energía disipada por el material del tope.

El objetivo es vincular datos experimentales de materiales (poliuretano) con modelos analíticos para optimizar el diseño y reducir pérdidas en planta.

## 🚀 Características Principales

* **Solver de Dinámica:** Resuelve la Ecuación Diferencial Ordinaria (EDO) del sistema utilizando el algoritmo **LSODA** (`scipy.integrate.odeint`), adecuado para sistemas rígidos y no rígidos.
* **Modelo de Impacto:** Implementa una función de fuerza de **pulso triangular** que simula el frenado brusco de la banda transportadora ($J = mv_0$).
* **Análisis Energético:** Calcula y visualiza el balance de energía en tiempo real, demostrando la conservación entre trabajo externo, energía cinética/potencial y calor disipado.
* **Optimización (Heatmap):** Genera mapas de calor para identificar la combinación óptima de rigidez ($k$) y amortiguamiento ($c$) que minimiza el desplazamiento.
* **Animación Didáctica:** Incluye un módulo de visualización (`visual.py`) que recrea el movimiento físico del tren de botellas y el amortiguador.

## 🛠️ Fundamento Matemático

La dinámica se rige por la segunda ley de Newton para un oscilador amortiguado forzado:

$$
m\ddot{x} + c\dot{x} + kx = F(t)
$$

Donde:
* $m$: Masa equivalente del tren de botellas (ej. 6 botellas de 3 L).
* $F(t)$: Pulso triangular de duración $\Delta t$ (50–100 ms).
* Condiciones iniciales: $x(0)=0,\ v(0)=0$ (reposo antes del impacto).

## 💻 Estructura del Proyecto

El repositorio contiene dos scripts principales para el análisis y la visualización:

1. `analisis.py`: Script de cálculo numérico. Genera las gráficas estáticas de respuesta temporal, balance de energía y mapa de calor de optimización.
2. `visual.py`: Script de animación. Muestra en tiempo real la compresión del resorte y el movimiento de las botellas sincronizado con las gráficas.

## 📊 Resultados Visuales

### Respuesta Dinámica y Balance de Energía

El sistema opera típicamente en régimen **subamortiguado** ($\zeta \approx 0.47$), logrando detener la carga en menos de 0.5 segundos con un desplazamiento controlado (~1.3 cm).

| Dinámica Temporal | Balance Energético |
|:---:|:---:|
| ![Dinámica](figura_1_respuesta_dinamica.png) | ![Energía](figura_2_balance_energia.png) |

### Optimización de Parámetros

El mapa de calor permite seleccionar materiales con la rigidez y amortiguamiento adecuados para minimizar el recorrido del tope.

![Heatmap](figura_3_heatmap_optimizacion.png)

## ⚙️ Requisitos e Instalación

Para ejecutar las simulaciones y la animación, necesitas Python 3.8+ y las librerías científicas estándar:

```bash
pip install numpy matplotlib scipy
