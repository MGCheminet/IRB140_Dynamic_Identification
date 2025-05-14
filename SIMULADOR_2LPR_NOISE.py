
def generateRandomDataSet(robot,N_segments,flag_joint_move=False,flag_plot=False,flag_q_noise=False, pos_noise_std=1e-6):
  """Genera un dataset para entrenamiento y pruebas

  Args:
    robot: Instancia del robot
    N_segments: Cantidad de puntos a visitar (fly by point). El movimiento empieza y termina con velocidad 0
    flag_joint_move: Indica si la secuencia de movimientos es joint o cartesiana
    flag_plot: Indica si se grafica la trayectoria deseada

  Returns:
    t_ref: Vector de tiempo de referencia
    q_ref: Vector de posiciones articulares de referencia
    qd_ref: Vector de velocidades articulares de referencia
    qdd_ref: Vector de aceleraciones articulares de referencia
    POSES_ref: Vector de posiciones cartesianas de referencia (Son las matrices A0_6 de cada posicion)
    tau_ref: Vector de torques de referencia calculado con la dinámica directa para realizar la trayectoria de referencia

  """
  alcanceXY = robot.a[0]+robot.a[1]
  Tj = np.random.rand(N_segments, )+0.5
  q_med = []
  if flag_joint_move:
    q_dest = (np.random.rand(N_segments, 2)-0.5)*2*np.pi
    t_ref,q_ref,qd_ref,qdd_ref,POSES_ref = robot.genTrJoint(q_dest,Tj)
    if flag_q_noise:
        #q_med = q_dest + np.random.normal(0, pos_noise_std, q_dest.shape)
        #t_med,q_ref,qd_ref,qdd_ref,POSES_ref = robot.genTrJoint(q_dest,Tj)
        q_med = q_ref + np.random.normal(0, pos_noise_std, q_ref.shape)
  else:
    # Limito las poses destino para trabajar en 2 cuadrantes y evitar pasar por singularidades
    POSE_dest = []
    while len(POSE_dest) < N_segments:
      x_dest = np.random.uniform(0, alcanceXY)
      y_dest = np.random.uniform(-alcanceXY, alcanceXY)
      if x_dest**2 + y_dest**2 < alcanceXY**2:
        POSE_dest.append(sm.SE3(x_dest, y_dest, 0))
    t_ref,q_ref,qd_ref,qdd_ref,POSES_ref = robot.genTrCart(POSE_dest,Tj)


  # Extraigo la posición del TCP para graficar
  pos_ref = np.vstack(([pose.t[0] for pose in POSES_ref], [pose.t[1] for pose in POSES_ref])).T

  # Obtengo la velocidad cartesiana derivando numéricamente
  posd_ref = np.diff(pos_ref, axis=0) / robot.Ts
  # Ajustar la longitud de qdd para que coincida con qd
  posd_ref = np.vstack([posd_ref, np.zeros(2,)])

  # Obtengo la aceleración cartesiana derivando numéricamente
  posdd_ref = np.diff(posd_ref, axis=0) / robot.Ts
  # Ajustar la longitud de qdd para que coincida con qd
  posdd_ref = np.vstack([posdd_ref, np.zeros(2,)])

  # Calculo con el PDI el torque para hacer esa trayectoria
  tau_ref = robot.rne(q_ref,qd_ref,qdd_ref)

  if flag_plot:
    # Graficar los resultados
    plt.figure(figsize=(12, 12))
    # Muestro las variables joint deseadas
    plt.subplot(3,1,1)
    plt.plot(t_ref,q_ref)
    plt.plot(t_ref,q_med)
    plt.legend(['q1', 'q2']);  plt.ylabel('q')
    plt.title('Variables articulares de referencia')
    plt.subplot(3,1,2)
    plt.plot(t_ref,qd_ref)
    plt.legend(['qd1', 'qd2']);  plt.ylabel('qd')
    plt.subplot(3,1,3)
    plt.plot(t_ref,qdd_ref)
    plt.legend(['qdd1', 'qdd2']); plt.xlabel('Tiempo'); plt.ylabel('qdd')
    plt.show()

    # Muestro las variables cartesianas deseadas
    plt.figure(figsize=(12, 12))
    plt.subplot(3,1,1)
    plt.plot(t_ref,pos_ref)
    plt.legend(['x', 'y']);  plt.ylabel('Posición')
    plt.title('Variables cartesianas de referencia')
    plt.subplot(3,1,2)
    plt.plot(t_ref,posd_ref)
    plt.legend(['vx', 'vy']);  plt.ylabel('Velocidad')
    plt.subplot(3,1,3)
    plt.plot(t_ref,posdd_ref)
    plt.legend(['a1', 'a2']); plt.xlabel('Tiempo'); plt.ylabel('Aceleración')
    plt.show()

    # Muestro la trayectoria deseada
    fig,ax = plt.subplots()
    plt.plot(pos_ref[:,0],pos_ref[:,1])
    circle = Circle((0, 0), alcanceXY,edgecolor='b', facecolor='none', linestyle='--')
    ax.add_patch(circle)
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(' Trayectoria de referencia')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.plot(t_ref,tau_ref)
    plt.legend(['tau1', 'tau2']); plt.xlabel('Tiempo'); plt.ylabel('Torque')
    plt.title('Torque calculado para realizar trayectoria')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(t_ref,q_ref, label='Q de Referencia')
    plt.plot(t_ref,q_med, label='Q Ruido')
    plt.legend(loc='upper right', fontsize=10)
    plt.xlabel('Tiempo'); plt.ylabel('Q')
    plt.title('Variables articulares')
    plt.show()

  return t_ref,q_ref,qd_ref,qdd_ref,POSES_ref,tau_ref, q_med

# %%
t_ref, q_ref, qd_ref, qdd_ref, POSES_ref, tau_ref, q_ref_noise = generateRandomDataSet(dp,50,flag_joint_move=True,flag_plot=True,flag_q_noise=True, pos_noise_std=1e-6)

# %%
nombre_archivo = f'./2lpr_noise_jointmovement_{fecha_actual}.csv'

cantidad = len(t_ref)

if not os.path.isfile(nombre_archivo) or os.path.getsize(nombre_archivo) == 0:
    # Si el archivo no existe o está vacío, escribir las cabeceras CSV
    with open(nombre_archivo, 'w') as archivo:
        archivo.write('ID, Time, Theta_1, Theta_2, D_Theta_1, D_Theta_2, DD_Theta_1, DD_Theta_2, Torque_1, Torque_2, Theta_N_1, Theta_N_2\n')
        #Length_1, Length_2, Alpha_1, Alpha_2 Mass_1, Mass_2, B_1, B_2, G_1, G_2
with open(nombre_archivo, 'a') as archivo:
    for _ in range(cantidad):
        id = _ +1
        time = t_ref[_]
        j_1 = q_ref[_][0]
        j_2 = q_ref[_][1]
        d_j_1 = qd_ref[_][0]
        d_j_2 = qd_ref[_][1]
        dd_j_1 = qdd_ref[_][0]
        dd_j_2 = qdd_ref[_][1]
        tau_1 = tau_ref[_][0]
        tau_2 = tau_ref[_][1]
        j_1_noise = q_ref_noise[_][0]
        j_2_noise = q_ref_noise[_][1]

        archivo.write(f'{id}, {time}, {j_1}, {j_2}, {d_j_1}, {d_j_2}, {dd_j_1}, {dd_j_2}, {tau_1}, {tau_2}, {j_1_noise}, {j_2_noise}\n')


