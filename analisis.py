"""
amortiguador_cinta_analisis.py
------------------------------
Análisis y simulación numérica de un tope viscoelástico para una línea de
embotellado, modelado como un sistema masa-resorte-amortiguador.

Este script genera 3 figuras estáticas para el análisis:
    • Figura 1: Gráficas de desplazamiento x(t) y velocidad v(t).
    • Figura 2: Desglose del balance de energía del sistema a lo largo del tiempo.
    • Figura 3: Mapa de calor (heatmap) del desplazamiento máximo en función de
                 los parámetros de rigidez (k) y amortiguamiento (c).

Requiere: Python ≥ 3.8, numpy, scipy, matplotlib.

Basado en el modelo de J. Muñoz Chávez
"""

# -------------------- 1. LIBRERÍAS --------------------
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

# -------------------- 2. PARÁMETROS DE SIMULACIÓN BASE --------------
# --- Parámetros Físicos ---
m = 5.0          # kg      (Masa equivalente del tren de 6 botellas de 3L)
k = 5000.0       # N/m     (Rigidez del tope de poliuretano)
c = 150.0        # N·s/m   (Coeficiente de amortiguamiento viscoso del tope)
v0 = 0.8         # m/s     (Velocidad inicial de las botellas al impactar)
dt_imp = 0.05    # s       (Duración del pulso de fuerza por frenado brusco)

# --- Parámetros Numéricos ---
t_sim = 2.0      # s       (Tiempo total de la ventana de simulación)
n_pts = 4000     #         (Número de puntos para la integración numérica)

# -------------------- 3. MODELO MATEMÁTICO --------------------
def fuerza_impacto(t: float, m_mass: float, v_init: float, t_pulse: float) -> float:
    """
    Define un pulso de fuerza triangular que modela el impacto.
    El área bajo la curva es el impulso total, J = m * v0.
    """
    if 0 <= t <= t_pulse:
        # Fórmula del pulso triangular que decae de 2J/Δt a 0
        return 2 * m_mass * v_init / t_pulse * (1 - t / t_pulse)
    return 0.0

def sistema_edo(y, t, m_mass, k_stiff, c_damp):
    """
    Define el sistema de Ecuaciones Diferenciales Ordinarias (EDO).
    y es el vector de estado [x, v], donde x es posición y v es velocidad.
    """
    x, v = y
    # Fuerza externa aplicada en el instante t
    f_ext = fuerza_impacto(t, m_mass, v0, dt_imp)
    
    # Ecuaciones de movimiento:
    # dx/dt = v
    # dv/dt = (F_externa - F_amortiguador - F_resorte) / m
    dxdt = v
    dvdt = (f_ext - c_damp * v - k_stiff * x) / m
    return [dxdt, dvdt]

# -------------------- 4. SIMULACIÓN DEL CASO BASE --------------------
print("1. Ejecutando simulación para el caso base...")
start_time = time.time()

# Vector de tiempo
t = np.linspace(0, t_sim, n_pts)
# Condiciones iniciales: x(0)=0, v(0)=0. La velocidad se adquiere por el pulso.
y0 = (0.0, 0.0)

# Resolver las EDO
sol = odeint(sistema_edo, y0, t, args=(m, k, c), atol=1e-9, rtol=1e-9)
x, v = sol[:, 0], sol[:, 1] # Extraer posición y velocidad

# --- Cálculos de Energía ---
Ep = 0.5 * k * x**2                       # Energía potencial elástica
Ec = 0.5 * m * v**2                       # Energía cinética
# La energía disipada es la integral de la potencia disipada (c*v^2)
dt = t[1] - t[0]
Ed = np.cumsum(c * v**2) * dt             # Integral numérica de la potencia disipada
F_ext_vec = np.array([fuerza_impacto(ti, m, v0, dt_imp) for ti in t])
W_ext = np.cumsum(F_ext_vec * v) * dt     # Trabajo hecho por la fuerza externa
Et = Ep + Ec                              # Energía mecánica total (sin contar disipación)

print(f"   Simulación base completada en {time.time() - start_time:.2f} s.")
print(f"   Desplazamiento máximo: {np.max(x)*100:.2f} cm")


# -------------------- 5. GRÁFICAS DE ANÁLISIS ESTÁTICO --------------------
print("2. Generando gráficas de análisis...")

# --- FIGURA 1: Posición y Velocidad ---
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(t, x * 100, label='Desplazamiento x(t) [cm]', color='dodgerblue', linewidth=2)
ax1.plot(t, v, label='Velocidad v(t) [m/s]', color='orangered', alpha=0.8)
ax1.set_xlabel('Tiempo [s]', fontsize=12)
ax1.set_ylabel('Magnitud', fontsize=12)
ax1.set_title('Respuesta Dinámica del Amortiguador', fontsize=14, fontweight='bold')
ax1.legend()
plt.tight_layout()
plt.savefig("figura_1_respuesta_dinamica.png", dpi=300)

# --- FIGURA 2: Balance de Energía ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(t, W_ext, label='Trabajo Externo (Entrada)', color='black', linewidth=2.5)
ax2.plot(t, Et + Ed, '--', label='Energía Total (Ep + Ec + E_disipada)', color='purple', linewidth=2)
ax2.plot(t, Ep, ':', label='Energía Potencial (Resorte)', color='green')
ax2.plot(t, Ec, ':', label='Energía Cinética', color='red')
ax2.plot(t, Ed, ':', label='Energía Disipada (Calor)', color='orange')
ax2.set_xlabel('Tiempo [s]', fontsize=12)
ax2.set_ylabel('Energía [J]', fontsize=12)
ax2.set_title('Balance Energético del Sistema', fontsize=14, fontweight='bold')
ax2.legend()
plt.tight_layout()
plt.savefig("figura_2_balance_energia.png", dpi=300)


# -------------------- 6. BARRIDO DE PARÁMETROS (k, c) --------------------
print("3. Realizando barrido de parámetros para heatmap (puede tardar)...")
start_time = time.time()

k_vals = np.linspace(2000, 8000, 31)      # Rango de rigidez
c_vals = np.linspace(50, 300, 31)        # Rango de amortiguamiento
xmax_map = np.zeros((len(c_vals), len(k_vals)))

for i, ci in enumerate(c_vals):
    for j, kj in enumerate(k_vals):
        # Resolver EDO para cada par (k, c)
        sol_s = odeint(sistema_edo, y0, t, args=(m, kj, ci), atol=1e-6, rtol=1e-6)
        # Guardar el desplazamiento máximo absoluto en cm
        xmax_map[i, j] = np.max(np.abs(sol_s[:, 0])) * 100

print(f"   Barrido completado en {time.time() - start_time:.2f} s.")

# -------------------- 7. GRÁFICA DE MAPA DE CALOR ----------------------
fig3, ax3 = plt.subplots(figsize=(9, 7))
# extent = [min_k, max_k, min_c, max_c]
im = ax3.imshow(xmax_map, origin='lower', aspect='auto',
                 extent=[k_vals[0]/1000, k_vals[-1]/1000, c_vals[0], c_vals[-1]],
                 cmap='viridis')
# Marcar el punto de los parámetros base
ax3.plot(k/1000, c, 'r*', markersize=12, label='Parámetros Base')
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Desplazamiento Máximo [cm]', fontsize=12)
ax3.set_xlabel('Rigidez k [kN/m]', fontsize=12)
ax3.set_ylabel('Amortiguamiento c [N·s/m]', fontsize=12)
ax3.set_title('Mapa de Optimización del Amortiguador', fontsize=14, fontweight='bold')
ax3.legend()
plt.tight_layout()
plt.savefig("figura_3_heatmap_optimizacion.png", dpi=300)

print("Análisis completo. Gráficas guardadas como archivos .png.")
plt.show()