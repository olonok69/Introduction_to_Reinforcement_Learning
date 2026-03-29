# Explicación de algoritmos de RL (con referencias al código)

Este documento explica cómo funcionan las implementaciones de los demos:

- Q-learning tabular en FrozenLake
- REINFORCE (gradiente de política) en CartPole
- PPO en CartPole (Stable-Baselines3)

Todas las explicaciones apuntan directamente al código.

Nota: los números de línea son aproximados y pueden cambiar conforme evolucionan los scripts; toma como referencia principal los nombres de funciones y bloques de código.

---

## 0) Plan de clase de 60 minutos (aprox.)

Usa esta distribución para mantener toda la explicación cerca de una hora:

- **0-8 min**: Fundamentos de RL (agente, entorno, estado, acción, recompensa, política).
- **8-15 min**: RL basado en modelo vs RL sin modelo (por qué estos demos usan métodos sin modelo).
- **15-30 min**: Q-learning (concepto + mapeo al código de FrozenLake).
- **30-45 min**: REINFORCE (concepto + mapeo al código de CartPole).
- **45-55 min**: PPO en práctica (flujo con Stable-Baselines3 + modos de evaluación).
- **55-60 min**: Cierre y preguntas, con comparación rápida: modelo vs política vs función de valor.

## 0.1 Fundamentos de RL desde la guía

Estos demos asumen los componentes estándar de RL destacados en `RL Guide.pdf`:

- **Agente**: el aprendiz/tomador de decisiones.
- **Entorno**: aquello con lo que interactúa el agente.
- **Estado** $s$: representación de la situación actual.
- **Acción** $a$: decisión disponible.
- **Recompensa** $r$: señal escalar de retroalimentación.
- **Política** $\pi$: mapeo de estados a acciones.

El objetivo es maximizar el retorno acumulado esperado con descuento a lo largo del tiempo.

## 0.2 Basado en modelo vs sin modelo (contexto)

- **RL basado en modelo** usa o aprende la dinámica de transición/recompensa para planificar.
- **RL sin modelo** aprende por prueba y error sin un modelo explícito del entorno.

Este repositorio se enfoca en demos **sin modelo** por simplicidad didáctica y de implementación:

- Q-learning (basado en valor, off-policy, tabular en este repositorio).
- REINFORCE y PPO (métodos de optimización de políticas).

## 1) Q-learning

### 1.1 Qué es Q-learning

Q-learning es un algoritmo de RL **sin modelo y basado en valor**. En lugar de aprender una política directamente, aprende una función $Q(s,a)$ que estima qué tan buena es cada acción en cada estado.

Idea central:

- Aprender valores de acción a partir de la interacción con el entorno.
- Derivar el comportamiento eligiendo la acción con mayor valor Q.

### 1.2 Cómo funciona Q-learning (conceptualmente)

En cada transición $(s,a,r,s')$, el algoritmo actualiza la estimación de $Q(s,a)$ usando un objetivo de Bellman:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

Bucle de alto nivel:

1. Seleccionar acción (normalmente con exploración `epsilon`-greedy).
2. Observar siguiente estado y recompensa.
3. Actualizar el valor Q actual hacia el objetivo TD.
4. Repetir durante muchos episodios, reduciendo la exploración con el tiempo.

### 1.3 Cómo funciona Q-learning en este proyecto (mapeo al código)

Archivo principal: `demos/q_learning_frozenlake.py`

### 1.3.1 Configuración del problema

- La creación del entorno y sus dimensiones está al inicio de `train(...)`.
- El agente usa una **tabla Q** inicializada en cero en ese mismo bloque de configuración.

Interpretación:

- `q_table[s, a]` guarda la estimación actual del retorno descontado esperado al tomar la acción `a` en el estado `s` y luego seguir la política codiciosa.

### 1.3.2 Selección de acción (`epsilon`-greedy)

Función: `epsilon_greedy_action(...)`

- Con probabilidad `epsilon`, elige una acción aleatoria (exploración).
- En caso contrario, elige `argmax_a Q(s,a)` (explotación).

Por qué importa:

- Al inicio se necesita exploración porque casi no hay conocimiento en la tabla Q.
- Más adelante conviene la explotación, y eso se logra con el decaimiento de epsilon.

### 1.3.3 Bucle de interacción por episodio

Bucle principal: bucles de episodio/paso dentro de `train(...)`

En cada paso:

1. Seleccionar acción con `epsilon_greedy_action(...)`.
2. Hacer `env.step` para obtener `(next_state, reward, terminated, truncated)`.
3. Marcar estado terminal con `done = terminated or truncated`.

### 1.3.4 Actualización de Bellman (regla de aprendizaje)

Bloque de actualización: líneas de objetivo/error TD dentro de `train(...)`

La implementación corresponde a:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

Mapeo con el código:

- `best_next = np.max(q_table[next_state])` es $\max_{a'}Q(s',a')$.
- `td_target = reward + gamma * best_next * (1-done)` es el objetivo.
- `td_error = td_target - q_table[state, action]`.
- `q_table[state, action] += alpha * td_error`.

Manejo de estados terminales:

- Multiplicar por `(1.0 - float(done))` evita el bootstrapping en transiciones terminales.

### 1.3.5 Decaimiento de exploración

Línea de actualización de epsilon dentro de `train(...)`:

$$
\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \cdot \epsilon_{decay})
$$

Esto mueve gradualmente el comportamiento de exploración a explotación.

### 1.3.6 Monitoreo y evaluación

- Registro de entrenamiento cada `log_every` episodios usando `avg100`.
- Evaluación final en `evaluate_policy(...)`:
  - Usa política puramente codiciosa (`argmax`).
  - Reporta tasa de éxito sobre muchos episodios.

---

## 2) REINFORCE

### 2.1 Qué es REINFORCE

REINFORCE es un algoritmo de **gradiente de política**. Optimiza directamente una política parametrizada $\pi_\theta(a|s)$, en lugar de aprender una tabla Q.

Idea central:

- Muestrear trayectorias con la política actual.
- Aumentar la probabilidad de acciones que produjeron alto retorno.
- Reducir la probabilidad de acciones que produjeron bajo retorno.

### 2.2 Cómo funciona REINFORCE (conceptualmente)

Para un episodio, calcula el retorno $G_t$ en cada paso y actualiza parámetros mediante ascenso por gradiente del retorno esperado; normalmente se implementa minimizando:

$$
\mathcal{L}(\theta) = -\sum_t \log \pi_\theta(a_t|s_t) G_t
$$

Bucle de alto nivel:

1. Ejecutar un episodio con la política estocástica actual.
2. Calcular retornos descontados.
3. Construir la pérdida de gradiente de política con log-probabilidades y retornos.
4. Hacer retropropagación y actualizar parámetros.

### 2.3 Cómo funciona REINFORCE en este proyecto (mapeo al código)

Archivo principal: `demos/reinforce_cartpole.py`

### 2.3.1 Red de política

Clase: `PolicyNetwork`

- Entrada: vector de observación (`obs_size`).
- Salida: probabilidades de acción con `softmax`.

Interpretación:

- El modelo representa una política estocástica $\pi_\theta(a|s)$.

### 2.3.2 Cálculo de retornos por trayectoria

Función: `discounted_returns(...)`

Para cada paso temporal `t`, el retorno es:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}
$$

Detalles de implementación:

- Calcula retornos hacia atrás.
- Normaliza retornos para reducir varianza del gradiente y estabilizar el entrenamiento.

### 2.3.3 Recolección de datos (on-policy)

Bucle de episodio: bucle de interacción dentro de `train(...)`

En cada paso:

1. Convertir observación a tensor.
2. Calcular probabilidades de acción con la política actual.
3. Muestrear acción con `Categorical(probs)`.
4. Guardar `log_prob(action)`.
5. Ejecutar acción y guardar recompensa.

Por qué el muestreo es importante:

- REINFORCE necesita muestras de la misma política que se está optimizando (on-policy Monte Carlo).

### 2.3.4 Pérdida de política y actualización

Bloque de pérdida: acumulación de pérdida de política dentro de `train(...)`

Este código implementa:

$$
\mathcal{L}(\theta) = -\sum_t \log \pi_\theta(a_t|s_t) G_t
$$

Interpretación:

- Si $G_t$ es alto, el ascenso por gradiente aumenta la probabilidad de la acción muestreada.
- Si $G_t$ es bajo, la actualización tiende a reducir esa probabilidad.

Paso de optimización:

- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`

### 2.3.5 Señal de convergencia

- Recompensa por episodio y promedio móvil (`avg100`) dentro de `train(...)`.
- Condición de “resuelto” con `avg100 >= solve_score` después de al menos 100 episodios.

---

## 3) Q-learning vs REINFORCE (resumen para clase)

### Qué se aprende

- Q-learning: aprende **valores estado-acción** `Q(s,a)` y deriva la política con argmax.
- REINFORCE: aprende la **política directamente** `\pi_\theta(a|s)`.

### Estilo de actualización

- Q-learning: bootstrapping por diferencia temporal (usa `max Q(s',a')`).
- REINFORCE: gradiente de política Monte Carlo (usa retornos de episodio completo).

### Dónde encaja mejor

- Q-learning (tabular): espacios de estado pequeños/discretos.
- REINFORCE (política neuronal): observaciones continuas o de mayor dimensión.

---

## 4) Hiperparámetros en estos scripts

### Q-learning (`parse_args(...)`)

- `alpha`: tasa de aprendizaje.
- `gamma`: factor de descuento.
- `epsilon-start`, `epsilon-min`, `epsilon-decay`: esquema de exploración.
- `episodes`, `max-steps`: presupuesto de entrenamiento.

### REINFORCE (`parse_args(...)`)

- `lr`: tasa de aprendizaje del optimizador.
- `gamma`: factor de descuento.
- `episodes`: máximo de episodios.
- `solve-score`: umbral para parada temprana.

---

## 5) Consejos prácticos para demos

- Si Q-learning se ve plano al inicio, aumenta episodios y/o desacelera `epsilon-decay`.
- Para clase, empieza con FrozenLake determinista (`--slippery` desactivado).
- Para REINFORCE, mantén la normalización de retornos y usa suficientes episodios.

---

## 6) Verificación postentrenamiento (flujo actualizado)

Los tres demos ahora incluyen modos de verificación postentrenamiento:

### 6.1 Render en tiempo real (`human`)

- Q-learning: `--render-eval --render-episodes N`
- REINFORCE: `--render-eval --render-episodes N`
- PPO: `--render-eval --render-episodes N`

Esto abre una ventana interactiva para verificar visualmente si la política aprendida resuelve la tarea.

### 6.2 Grabación de video (`mp4` para diapositivas)

- Q-learning: `--record-video --video-episodes N --video-dir <path>`
- REINFORCE: `--record-video --video-episodes N --video-dir <path>`
- PPO: `--record-video --video-episodes N --video-dir <path>`

Esto graba episodios de evaluación con Gymnasium `RecordVideo` en:

- `videos/q_learning_frozenlake` (predeterminado)
- `videos/reinforce_cartpole` (predeterminado)
- `videos/ppo_cartpole` (predeterminado)

### 6.3 Modo combinado (un solo comando)

- Q-learning: `--record-and-render`
- REINFORCE: `--record-and-render`
- PPO: `--record-and-render`

Esto activa ambos modos de visualización en una sola ejecución.

### 6.4 Nota de dependencias

- La grabación de video requiere `moviepy` (ya agregado a `requirements.txt`).
- Si falta, los scripts muestran una ayuda clara: `uv pip install moviepy`.

---

## 7) PPO

### 7.1 Qué es PPO

PPO (Proximal Policy Optimization) es un algoritmo moderno de **optimización de políticas** diseñado para mejorar estabilidad y eficiencia de muestras frente a métodos de gradiente de política más básicos.

Idea central:

- Optimizar un objetivo sustituto que desincentiva actualizaciones demasiado grandes.
- Reusar datos de trayectorias durante varios epochs de optimización.
- Balancear desempeño y estabilidad de entrenamiento.

En este proyecto, PPO viene de `stable-baselines3`.

### 7.2 Cómo funciona PPO (conceptualmente)

PPO recolecta rollouts, calcula estimaciones de ventaja y actualiza redes de política/valor con un objetivo con clipping para evitar cambios bruscos de política en un solo paso.

Bucle de alto nivel:

1. Recolectar datos de rollout con la política actual.
2. Estimar ventajas/retornos.
3. Ejecutar varias actualizaciones por gradiente en minibatches con clipping.
4. Repetir hasta consumir el presupuesto de timesteps.

### 7.3 Cómo funciona PPO en este proyecto (mapeo al código)

Archivo principal: `demos/gymnasium_ppo_cartpole.py`

### 7.3.1 Configuración de entrenamiento

- `train_and_evaluate` crea `CartPole-v1` con el wrapper `Monitor` y construye `PPO` (inicio de función + bloque del constructor).
- El script usa `MlpPolicy` y los hiperparámetros principales desde CLI: `learning_rate`, `n_steps`, `batch_size`, `gamma` y `seed`.

Interpretación práctica:

- PPO es un algoritmo de optimización de políticas que mantiene actualizaciones estables al limitar cambios bruscos en la política (implementado internamente por Stable-Baselines3).

### 7.3.2 Entrenar, guardar y evaluar

- El entrenamiento se ejecuta con `model.learn(total_timesteps=args.timesteps)`.
- El guardado del modelo se hace con `model.save(args.model_path)` después de asegurar que exista el directorio padre.
- La evaluación determinista usa `evaluate_policy(..., deterministic=True)` y reporta recompensa media/desviación estándar.

Esto da una métrica clara de “¿aprendió o no?” justo después del entrenamiento.

### 7.3.3 Verificación visual y grabación

- Ruta de render humano: `render_policy(...)` con `render_mode="human"`, acciones deterministas con `model.predict(..., deterministic=True)`.
- Ruta de grabación en video: `record_policy_video(...)` con `render_mode="rgb_array"` y wrapper `RecordVideo` de Gymnasium.
- Ruta combinada: `--record-and-render` activa render y grabación en una sola ejecución.

### 7.3.4 Parámetros CLI a destacar en clase

- `--timesteps`: presupuesto total de entrenamiento.
- `--learning-rate`, `--n-steps`, `--batch-size`, `--gamma`: controles centrales de PPO.
- `--eval-episodes`: confiabilidad de la estimación de evaluación.
- `--render-eval`, `--record-video`, `--record-and-render`: modos de verificación de política postentrenamiento.

---

## 8) Modelo vs política vs función de valor (aclaración rápida)

Desde la guía, estos tres conceptos responden preguntas distintas:

- **Modelo**: “¿Qué va a pasar?”
  - Predice siguiente estado/recompensa; se usa en RL basado en modelo.
- **Política**: “¿Qué debo hacer ahora?”
  - Elige acciones a partir del estado (determinística o estocástica).
- **Función de valor**: “¿Qué tan bueno es esto?”
  - Estima el retorno esperado futuro de un estado $V(s)$ o de un par estado-acción $Q(s,a)$.

En estos demos:

- Q-learning aprende principalmente una **función de valor** (`Q-table`) y deriva la política con argmax.
- REINFORCE/PPO optimizan principalmente una **política**, y PPO además usa estimación de valor internamente.
