"""
amortiguador_cinta_visualizacion.py
------------------------------------
Visualización dinámica y didáctica de la simulación del amortiguador de la
cinta transportadora.

El script realiza lo siguiente:
1. Resuelve numéricamente el sistema con los parámetros definidos.
2. Crea una animación que muestra:
   - Una vista esquemática del tren de botellas, el resorte y el amortiguador.
   - Gráficas en tiempo real de desplazamiento y velocidad.
   - Un indicador de la fuerza de impacto.

Es ideal para presentaciones y para entender intuitivamente el comportamiento
del sistema.

Requiere: Python ≥ 3.8, numpy, scipy, matplotlib.
"""

# -------------------- 1. LIBRERÍAS --------------------
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# -------------------- 2. PARÁMETROS Y MODELO --------------------
# (Son los mismos que en el script de análisis)
m = 5.0
k = 5000.0
c = 150.0
v0 = 0.8
dt_imp = 0.05
t_sim = 2.0
n_pts = 500 # Menos puntos para una animación más fluida

def fuerza_impacto(t: float, m_mass: float, v_init: float, t_pulse: float) -> float:
    if 0 <= t <= t_pulse:
        return 2 * m_mass * v_init / t_pulse * (1 - t / t_pulse)
    return 0.0

def sistema_edo(y, t, m_mass, k_stiff, c_damp):
    x, v = y
    f_ext = fuerza_impacto(t, m_mass, v0, dt_imp)
    dxdt = v
    dvdt = (f_ext - c_damp * v - k_stiff * x) / m
    return [dxdt, dvdt]

# -------------------- 3. RESOLVER EDO (UNA SOLA VEZ) --------------------
print("Resolviendo la trayectoria para la animación...")
t = np.linspace(0, t_sim, n_pts)
y0 = (0.0, 0.0)
sol = odeint(sistema_edo, y0, t, args=(m, k, c))
x_sol, v_sol = sol[:, 0], sol[:, 1]
f_sol = np.array([fuerza_impacto(ti, m, v0, dt_imp) for ti in t])
print("¡Listo para animar!")

# -------------------- 4. CONFIGURACIÓN DE LA VISUALIZACIÓN --------------------
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 2) # 3 filas, 2 columnas

# --- Ejes de la figura ---
ax_anim = fig.add_subplot(gs[0, :])  # Eje para la animación (fila superior, ancho completo)
ax_pos = fig.add_subplot(gs[1, 0])   # Eje para gráfica de posición
ax_vel = fig.add_subplot(gs[1, 1])   # Eje para gráfica de velocidad
ax_force = fig.add_subplot(gs[2, :]) # Eje para la fuerza de impacto

# --- Configuración del eje de animación (ax_anim) ---
ax_anim.set_xlim(-0.02, np.max(x_sol) * 1.2)
ax_anim.set_ylim(-1, 1)
ax_anim.set_yticks([]) # Ocultar eje Y
ax_anim.set_xlabel("Posición [m]")
ax_anim.set_title("Simulación del Amortiguador de la Cinta Transportadora", fontweight='bold')
ax_anim.axvline(0, color='black', linestyle='--', label='Pared Fija')

# Elementos estáticos de la animación
pared = ax_anim.plot([-0.002, -0.002], [-1, 1], color='gray', linewidth=10)
cinta = ax_anim.add_patch(patches.Rectangle((-0.02, -0.6), 
                            width=np.max(x_sol)*1.3, height=0.1, color='silver'))

# Elementos dinámicos (que se actualizarán en cada frame)
ancho_botellas = 0.01
botellas = ax_anim.add_patch(patches.Rectangle((0, -0.5), ancho_botellas, 1, 
                                               fc='saddlebrown', ec='black', label='Tren de Botellas'))
spring, = ax_anim.plot([], [], 'k-', lw=2)
damper_piston, = ax_anim.plot([], [], 'k-', lw=4)
damper_cilindro, = ax_anim.plot([], [], 'k-', lw=8, alpha=0.5)
tiempo_texto = ax_anim.text(0.95, 0.9, '', transform=ax_anim.transAxes, ha='right', fontsize=12)


# --- Configuración de los ejes de gráficas ---
def setup_graph_axes(ax, title, ylabel, x_data, y_data):
    ax.set_xlim(0, t_sim)
    ax.set_ylim(np.min(y_data) * 1.2, np.max(y_data) * 1.2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    line, = ax.plot([], [], lw=2)
    return line

line_pos = setup_graph_axes(ax_pos, "Desplazamiento", "x [m]", t, x_sol)
line_vel = setup_graph_axes(ax_vel, "Velocidad", "v [m/s]", t, v_sol)
line_force = setup_graph_axes(ax_force, "Fuerza de Impacto", "Fuerza [N]", t, f_sol)
ax_force.set_xlabel("Tiempo [s]")


# -------------------- 5. FUNCIONES DE ANIMACIÓN --------------------
def init():
    """Función de inicialización para la animación."""
    botellas.set_x(0)
    spring.set_data([], [])
    damper_piston.set_data([], [])
    damper_cilindro.set_data([], [])
    line_pos.set_data([], [])
    line_vel.set_data([], [])
    line_force.set_data([], [])
    tiempo_texto.set_text('')
    return (botellas, spring, damper_piston, damper_cilindro, 
            line_pos, line_vel, line_force, tiempo_texto)

def get_spring_points(x_pos, start=0, n_coils=10):
    """Calcula las coordenadas para dibujar un resorte comprimido/estirado."""
    x = np.array([start, *(np.linspace(start, x_pos, n_coils*2+1)) , x_pos])
    y = np.array([0, *([0.2, -0.2]*n_coils), 0, 0])
    return x, y

def animate(i):
    """Función que se llama en cada frame de la animación."""
    # Posición actual del tren de botellas
    x_i = x_sol[i]
    
    # Actualizar la animación
    botellas.set_x(x_i)
    spring_x, spring_y = get_spring_points(x_i, n_coils=8)
    spring.set_data(spring_x, spring_y*0.5 - 0.2) # Mover el resorte abajo
    
    # Dibujar el amortiguador
    damper_piston.set_data([x_i-0.005, 0], [0.2, 0.2])
    damper_cilindro.set_data([x_i, x_i+ancho_botellas], [0.2, 0.2])
    
    # Actualizar las gráficas
    line_pos.set_data(t[:i+1], x_sol[:i+1])
    line_vel.set_data(t[:i+1], v_sol[:i+1])
    line_force.set_data(t[:i+1], f_sol[:i+1])
    
    # Actualizar el texto del tiempo
    tiempo_texto.set_text(f'Tiempo = {t[i]:.2f} s')
    
    return (botellas, spring, damper_piston, damper_cilindro, 
            line_pos, line_vel, line_force, tiempo_texto)

# -------------------- 6. EJECUTAR LA ANIMACIÓN --------------------
# blit=True significa que solo se redibujan las partes que han cambiado.
ani = animation.FuncAnimation(fig, animate, frames=len(t),
                              init_func=init, blit=True, interval=t_sim, repeat=True)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para el título principal
plt.show()