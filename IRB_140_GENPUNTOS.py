# %%
import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
import pandas as pd #Lo necesito para cargar y leer el csv

import os
import random
import time
from datetime import datetime

import tensorflow
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential #



# %%
#Parametros DH del IRB140 Verificando segun robotstudio
a_irb140 = np.array([70, 360., 0, 0., 0, 0])
d_irb140 = np.array([352., 0, 0, 380, 0, 0])

#a_irb140 = np.array([70, 360., 0, 0., 0, 0])
#d_irb140 = np.array([0, 0, 0, 732, 0, -65])
alpha_irb140 = np.array([-1, 0, -1, 1, -1, 0]) * np.pi / 2.


# theta_irb140_real = theta_irb140 + thetas_correction
thetas_correction = np.array([0, -90, 0, 0, 0, -180]) * (np.pi/180)
theta_irb140 = np.array([52.8, -2, 10, 13, 56, 28]) * (np.pi/180) #Estos thetas son para probar

# %%
theta = smp.symbols('theta1:{0}'.format(7), real=True)
a = smp.symbols('a1:{0}'.format(7), real=True)
d = smp.symbols('d1:{0}'.format(7), real=True)
alpha = smp.symbols('alpha1:{0}'.format(7), real=True)

# %%
#Problema cinemático directo
def symbolic_irb140_forward_kinematics():
  J = []
  for i in range(6):
      # Use symbolic expressions for DH parameters
      A_i = smp.Matrix([
          [smp.cos(theta[i]), -smp.sin(theta[i]) * smp.cos(alpha[i]), smp.sin(theta[i]) * smp.sin(alpha[i]), a[i] * smp.cos(theta[i])],
          [smp.sin(theta[i]), smp.cos(theta[i]) * smp.cos(alpha[i]), -smp.cos(theta[i]) * smp.sin(alpha[i]), a[i] * smp.sin(theta[i])],
          [0.0, smp.sin(alpha[i]), smp.cos(alpha[i]), d[i]],
          [0.0, 0.0, 0.0, 1.0]
      ])
      J.append(A_i)
      J[i] = smp.simplify(J[i])

  A_0_n = J[0]
  for matrix in range(5):
    A_0_n = A_0_n * J[matrix+1]

  #A_0_n
  J.append(A_0_n)
  return J

# %%
def irb140_forward_kinematics(J, thetas):
    sym_expressions = [
      [expr for expr in row] for row in J
    ]

    lambdified_expr = np.array([
          [smp.lambdify((a, d, alpha, theta), expr, 'numpy') for expr in J[i]]
          for i in range(6+1)
      ])

    matrices = []
    for row in lambdified_expr:
        T = np.block([expr(a_irb140, d_irb140, alpha_irb140, thetas) for expr in row])
        matrices.append(T)
    replaced_matrixes = np.array(matrices)
    replaced_matrixes = np.asarray(replaced_matrixes).reshape(replaced_matrixes.shape[0], 4, 4)
    return replaced_matrixes

# %%
J_sym = symbolic_irb140_forward_kinematics()
J_matrix = irb140_forward_kinematics(J_sym, theta_irb140+thetas_correction)
J_06 = irb140_forward_kinematics(J_sym, theta_irb140+thetas_correction)[-1,0:3,3]
J_06#Funciona con la verificación ya que para el theta de prueba me da lo mismo q en robotstudio array([269.92581072, 375.66383285, 601.84099693])

# %%
#print(smp.latex(J_sym[6]))
symbolic_constants = {
    smp.symbols('alpha1', real=True): alpha_irb140[0],
    smp.symbols('alpha2', real=True): alpha_irb140[1],
    smp.symbols('alpha3', real=True): alpha_irb140[2],
    smp.symbols('alpha4', real=True): alpha_irb140[3],
    smp.symbols('alpha5', real=True): alpha_irb140[4],
    smp.symbols('alpha6', real=True): alpha_irb140[5],
    smp.symbols('a3', real=True): a_irb140[2],
    smp.symbols('a4', real=True): a_irb140[3],
    smp.symbols('a5', real=True): a_irb140[4],
    smp.symbols('a6', real=True): a_irb140[5],
    smp.symbols('d2', real=True): d_irb140[1],
    smp.symbols('d3', real=True): d_irb140[2],
    smp.symbols('d5', real=True): d_irb140[4],
    smp.symbols('d6', real=True): d_irb140[5],
    smp.symbols('theta2', real=True): smp.symbols('theta2', real=True) - (90 * smp.pi/180),
}

# Realizar las sustituciones en la expresión simbólica
J_sym_substituted = [expr.subs(symbolic_constants) for expr in J_sym[6]]
J_sym_substituted = [expr.rewrite(smp.sin, smp.cos) for expr in J_sym_substituted]
J_sym_substituted = [smp.nsimplify(expr, tolerance=1e-10) for expr in J_sym_substituted]


J_sym_substituted[0]

# %%
smp.trigsimp(J_sym_substituted[3])

# %%
J_sym[0]

# %%
px_s = smp.symbols('px', real=True)
py_s = smp.symbols('py', real=True)
pz_s = smp.symbols('pz', real=True)

point_matrix = smp.Matrix(4,1,[px_s, py_s, pz_s, 1])

point_matrix

# %%
J_01_sym_substituted = [expr.subs(symbolic_constants) for expr in J_sym[0]]
J_01_sym_substituted = [expr.rewrite(smp.sin, smp.cos) for expr in J_01_sym_substituted]
J_01_sym_substituted = [smp.nsimplify(expr, tolerance=1e-10) for expr in J_01_sym_substituted]
J_01_sym_substituted = [smp.trigsimp(expr) for expr in J_01_sym_substituted]
J_01_sym_substituted =  smp.Matrix(4, 4, J_01_sym_substituted)
smp.trigsimp(J_01_sym_substituted.inv()*point_matrix)

# %%
J_12_sym_substituted = [expr.subs(symbolic_constants) for expr in J_sym[1]]
J_12_sym_substituted = [expr.rewrite(smp.sin, smp.cos) for expr in J_12_sym_substituted]
J_12_sym_substituted = [smp.nsimplify(expr, tolerance=1e-10) for expr in J_12_sym_substituted]
J_12_sym_substituted = [smp.trigsimp(expr) for expr in J_12_sym_substituted]
J_12_sym_substituted =  smp.Matrix(4, 4, J_12_sym_substituted)
J_12_sym_substituted

# %%
J_23_sym_substituted = [expr.subs(symbolic_constants) for expr in J_sym[2]]
J_23_sym_substituted = [expr.rewrite(smp.sin, smp.cos) for expr in J_23_sym_substituted]
J_23_sym_substituted = [smp.nsimplify(expr, tolerance=1e-10) for expr in J_23_sym_substituted]
J_23_sym_substituted = [smp.trigsimp(expr) for expr in J_23_sym_substituted]
J_23_sym_substituted =  smp.Matrix(4, 4, J_23_sym_substituted)
J_23_sym_substituted

# %%
J_34_sym_substituted = [expr.subs(symbolic_constants) for expr in J_sym[3]]
J_34_sym_substituted = [expr.rewrite(smp.sin, smp.cos) for expr in J_34_sym_substituted]
J_34_sym_substituted = [smp.nsimplify(expr, tolerance=1e-10) for expr in J_34_sym_substituted]
J_34_sym_substituted = [smp.trigsimp(expr) for expr in J_34_sym_substituted]
J_34_sym_substituted =  smp.Matrix(4, 4, J_34_sym_substituted)
J_34_sym_substituted[:,3]

# %%
J_12_sym_substituted*J_23_sym_substituted*J_34_sym_substituted[:,3]
#smp.trigsimp(J_12_sym_substituted*J_23_sym_substituted*J_34_sym_substituted[:,3])

# %%
def irb140_inverse_kinematics_arm(point):
  px = point[0]
  py = point[1]
  pz = point[2]

  q1 = np.arctan2((py/(np.sqrt(px**2+py**2)**2)),(px/(np.sqrt(px**2+py**2)**2)))#q1 = q1 * 180/np.pi #Para devolver en grados
  c1 = np.cos(q1) #los cosenos y senos van en radianes
  s1 = np.sin(q1)

  s3 = ((px*c1+py*s1-a_irb140[0])**2 + (d_irb140[0]-pz)**2 - (d_irb140[3]**2+a_irb140[1]**2)) / (2*a_irb140[1]*d_irb140[3])
  q3 = np.arctan2(-s3,np.sqrt(1-s3**2)) #+ np.pi
  c3 = np.cos(q3)

  c2 = ((px*c1+py*s1-a_irb140[0])*(c3*d_irb140[3])-(d_irb140[0]-pz)*(a_irb140[1]+s3*d_irb140[3])) / ((a_irb140[1]+s3*d_irb140[3])**2+(c3*d_irb140[3])**2)
  s2 = ((px*c1+py*s1-a_irb140[0])*(a_irb140[1]+s3*d_irb140[3])+(d_irb140[0]-pz)*(c3*d_irb140[3])) / ((a_irb140[1]+s3*d_irb140[3])**2+(c3*d_irb140[3])**2)
  q2 = np.arctan2(s2,c2)

  return np.array([q1, q2, q3])

# %%
#Reescribir porque
def irb140_inverse_kinematics(J_):
  J_0_6 = J_[-1,0:3,3]
  px = J_0_6[0]
  py = J_0_6[1]
  pz = J_0_6[2]

  q1 = np.arctan2((py/(np.sqrt(px**2+py**2)**2)),(px/(np.sqrt(px**2+py**2)**2)))#q1 = q1 * 180/np.pi #Para devolver en grados
  c1 = np.cos(q1) #los cosenos y senos van en radianes
  s1 = np.sin(q1)

  s3 = ((px*c1+py*s1-a_irb140[0])**2 + (d_irb140[0]-pz)**2 - (d_irb140[3]**2+a_irb140[1]**2)) / (2*a_irb140[1]*d_irb140[3])
  q3 = np.arctan2(-s3,np.sqrt(1-s3**2)) #+ np.pi
  c3 = np.cos(q3)

  c2 = ((px*c1+py*s1-a_irb140[0])*(c3*d_irb140[3])-(d_irb140[0]-pz)*(a_irb140[1]+s3*d_irb140[3])) / ((a_irb140[1]+s3*d_irb140[3])**2+(c3*d_irb140[3])**2)
  s2 = ((px*c1+py*s1-a_irb140[0])*(a_irb140[1]+s3*d_irb140[3])+(d_irb140[0]-pz)*(c3*d_irb140[3])) / ((a_irb140[1]+s3*d_irb140[3])**2+(c3*d_irb140[3])**2)
  q2 = np.arctan2(s2,c2) #+ np.pi/2

  #Lo que queda despejar es la rototraslacion R_3_6(q4, q5, q6)
  #R_01 = J_[0,0:3,0:3]
  #R_12 = J_[1,0:3,0:3]
  #R_23 = J_[2,0:3,0:3]
  R_34 = J_[3,0:3,0:3]
  R_45 = J_[4,0:3,0:3]
  R_56 = J_[5,0:3,0:3]
  #R_06 = J_[6,0:3,0:3]

  R_36 = np.dot(np.dot(R_34,R_45),R_56)

  n_z_apos = R_36[2,0]
  s_z_apos = R_36[2,1]
  a_x_apos = R_36[0,2]
  a_y_apos = R_36[1,2]
  a_z_apos = R_36[2,2]

  q5 = np.arctan2(np.sqrt(1-a_z_apos**2),a_z_apos)
  s5 = np.sin(q5)

  q4 = np.arctan2(-a_y_apos/s5,-a_x_apos/s5)
  q6 = np.arctan2(s_z_apos/s5,-n_z_apos/s5)

  thetas = np.array([q1, q2, q3, q4, q5, q6])
  return thetas

# %%
#np.array([52.8, -2, 10, 13, 56, 28])
(irb140_inverse_kinematics(J_matrix)) * 180/np.pi

# %%
(irb140_inverse_kinematics_arm(J_matrix[-1,0:3,3])) * 180/np.pi

# %%
def define_line_3d(point_1, point_2):
    # Encontrar el vector director de la línea
    direction = point_2 - point_1
    distance = np.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2 + (point_2[2] - point_1[2])**2)

    # Crear puntos a lo largo de la línea
    t = np.linspace(0, 1, 1000)
    line_points = np.array([point_1 + t_val * direction for t_val in t])
    return line_points

# %% [markdown]
# # **Creación de Archivo de Poses IRB-140**
# Se generarán poses aleatorias dentro de los rangos de cada joint
# %%
def check_invalid_trajectory(trajectory, center, radius):
  invalid_intersection = False
  center_array = np.array(center)
  radius_squared = radius ** 2
  for point in trajectory:
      if np.sum((np.array(point) - center_array) ** 2) <= radius_squared:
          return True
  return False

def check_joint_range(trajectory, min_joints, max_joints):
  for point in trajectory:
    ibk = irb140_inverse_kinematics_arm(np.array(point))
    if (ibk[0] >= (max_joints[0] * np.pi/180))|(ibk[1] >= (max_joints[1] * np.pi/180))|(ibk[2] >= (max_joints[2] * np.pi/180)):
          return True
    #if (ibk[0] < (min_joints[0] * np.pi/180))|(ibk[1] < (min_joints[1] * np.pi/180))|(ibk[2] < (min_joints[2] * np.pi/180)):
    #      return True
  return False



def generar_angulos_validos(min_joints, max_joints, cantidad, thetas_correction):
    J_sym = symbolic_irb140_forward_kinematics()
    j_1 = []
    j_2 = []
    j_3 = []
    j_4 = []
    j_5 = []
    j_6 = []
    tcp_point = []

    id = 0
    while id < cantidad:
        thetas = np.array([random.uniform(min_joints[i], max_joints[i]) for i in range(6)]) * np.pi/180
        matrixes = irb140_forward_kinematics(J_sym, thetas + thetas_correction)
        tcp_point_actual = matrixes[-1, 0:3, 3]

        if id > 0:
            trajectory = define_line_3d(tcp_point[id-1], tcp_point_actual)
            #if check_invalid_trajectory(trajectory, [70, 0, 352], 350):
            #   continue
            if check_joint_range(trajectory,min_joints[0:3], max_joints[0:3]):
                continue

        j_1.append(thetas[0] * 180/np.pi)
        j_2.append(thetas[1] * 180/np.pi)
        j_3.append(thetas[2] * 180/np.pi)
        j_4.append(thetas[3] * 180/np.pi)
        j_5.append(thetas[4] * 180/np.pi)
        j_6.append(thetas[5] * 180/np.pi)
        tcp_point.append(tcp_point_actual)
        id += 1

    return j_1, j_2, j_3, j_4, j_5, j_6

# %%
min_joints = [0, -12, -90, 0, 0, 0]
min_joints[0:3]

# %%
#len(j__1)
J_sym = symbolic_irb140_forward_kinematics()
thetas_p1 = np.array([57.168, 12.134, -82.731, 20.735, 74.071, 43.511]) * np.pi/180
thetas_p2 = np.array([75.240, 68.669, 30.014, 86.986, 25.097, 25.771]) * np.pi/180

tcp_point_p1 = irb140_forward_kinematics(J_sym, thetas_p1 + thetas_correction)[-1,0:3,3]
tcp_point_p2 = irb140_forward_kinematics(J_sym, thetas_p2 + thetas_correction)[-1,0:3,3]

trajectory = define_line_3d(tcp_point_p1, tcp_point_p2)

#question_valid = check_invalid_trajectory(trajectory, [70, 0, 352], 500)

question_valid = check_joint_range(trajectory,[0, -12, -90], [90, 110, 50])

question_valid

# %%
irb140_forward_kinematics(J_sym, thetas_p1 + thetas_correction)[-1,0:3,3] #array([171.6084395 , 136.84885328, 803.47491906])

# %%
tcp_point_p1

# %%
#El área esférica está centrada en (70, yc, 352) el radio es 254 (voy a tomar y=0 porque no tengo un valor exacto)
#Los puntos para definir la esfera son (314, y1, 421), (1, y2, 596), (218, y3, 558)

# Función para generar números aleatorios dentro de un rango y guardarlos en un archivo CSV
def generar_valores_IRB140(v_ranges, z_ranges, min_joints, max_joints, cantidad, thetas_correction, nombre_archivo):
    directorio = os.path.dirname(nombre_archivo)
    os.makedirs(directorio, exist_ok=True)
    
    if not os.path.isfile(nombre_archivo) or os.path.getsize(nombre_archivo) == 0:
        # Si el archivo no existe o está vacío, escribir las cabeceras CSV
        with open(nombre_archivo, 'w') as archivo:
            archivo.write('ID, Vel TCP, Pres, Theta_1, Theta_2, Theta_3, Theta_4, Theta_5, Theta_6\n')
    with open(nombre_archivo, 'a') as archivo:
        j_1_array, j_2_array, j_3_array, j_4_array, j_5_array, j_6_array = generar_angulos_validos(min_joints, max_joints, cantidad, thetas_correction)
        for i in range(cantidad):
            id = i
            vel = random.uniform(v_ranges[0], v_ranges[1])
            z = random.uniform(z_ranges[0], z_ranges[1])
            j_1 = j_1_array[i]
            j_2 = j_2_array[i]
            j_3 = j_3_array[i]
            j_4 = j_4_array[i]
            j_5 = j_5_array[i]
            j_6 = j_6_array[i]

            archivo.write(f'{id}, {vel}, {z}, {j_1}, {j_2}, {j_3}, {j_4}, {j_5}, {j_6}\n')


#Rangos de velocidad en mm/s
v_ranges = [10, 200]

#Rangos de presicion en mm
z_ranges = [10, 30]

# Cantidad de números aleatorios a generar
cantidad = 1000
fecha_actual = datetime.now().strftime("%Y-%m-%d")
calculate = False #True
not_limited = False

if calculate:
  if not_limited:
    #Rangos para cada joint -> Completos
    min_joints = [-180, -90, -230, -200, -115, -360]
    max_joints = [180, 110, 50, 200, 115, 360]

    # Nombre del archivo donde se guardarán los ángulos
    nombre_archivo = f'./Pruebas_Iniciales/datos_irb140_{fecha_actual}.csv'
  else:
    #Rangos para cada joint -> Limitados a robconf [0 0 0 0] siendo m incoginita pero m = 0 (z > 0 segun grafico IRB 140 Product Specification)
    min_joints = [0, -12, -90, 0, 0, 0]
    max_joints = [90, 110, 50, 90, 110, 90]

    # Nombre del archivo donde se guardarán los ángulos
    nombre_archivo = f'./Pruebas_Iniciales/datos_irb140_{fecha_actual}_LIMITED_2.csv'


  # Llamada a la función para generar y guardar los ángulos
  generar_valores_IRB140(v_ranges, z_ranges, min_joints, max_joints, cantidad, thetas_correction, nombre_archivo)

  print(f'Se han añadido {cantidad} en el archivo "{nombre_archivo}".')
