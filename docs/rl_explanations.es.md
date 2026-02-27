# Explicación de Algoritmos de RL (con referencias al código)

Este documento explica cómo funcionan las implementaciones de los demos:

- Q-learning tabular en FrozenLake
- REINFORCE (Policy Gradient) en CartPole
- PPO en CartPole (Stable-Baselines3)

Todas las explicaciones apuntan directamente al código.

Nota: los números de línea son aproximados y pueden cambiar cuando evolucionan los scripts; toma como referencia principal los nombres de funciones y bloques de código.

---

## 1) Q-learning

### 1.1 Qué es Q-learning

Q-learning es un algoritmo de reinforcement learning **model-free y value-based**. En lugar de aprender una policy directamente, aprende una función $Q(s,a)$ que estima qué tan buena es cada acción en cada estado.

Idea central:

- Aprender action values desde la interacción con el environment.
- Derivar el comportamiento eligiendo la acción con mayor Q-value.

### 1.2 Cómo funciona Q-learning (conceptualmente)

En cada transición $(s,a,r,s')$, el algoritmo actualiza la estimación de $Q(s,a)$ usando un Bellman target:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

Loop de alto nivel:

1. Seleccionar acción (normalmente con exploration `epsilon`-greedy).
2. Observar siguiente estado y reward.
3. Actualizar el Q-value actual hacia el TD target.
4. Repetir por muchos episodios, reduciendo exploration con el tiempo.

### 1.3 Cómo funciona Q-learning en este proyecto (mapeo al código)

Archivo principal: `demos/q_learning_frozenlake.py`

### 1.3.1 Configuración del problema

- La creación del environment y sus dimensiones está al inicio de `train(...)`.
- El agente usa una **Q-table** inicializada en cero en ese mismo bloque de setup.

Interpretación:

- `q_table[s, a]` guarda la estimación actual del retorno descontado esperado al tomar la acción `a` en el estado `s` y luego seguir la policy greedy.

### 1.3.2 Selección de acción (`epsilon`-greedy)

Función: `epsilon_greedy_action(...)`

- Con probabilidad `epsilon`, elige una acción aleatoria (exploration).
- En caso contrario, elige `argmax_a Q(s,a)` (exploitation).

Por qué importa:

- Al inicio se necesita exploration porque casi no hay conocimiento en la Q-table.
- Más adelante conviene exploitation, y eso se logra con epsilon decay.

### 1.3.3 Loop de interacción por episodio

Loop principal: loops de episodio/step dentro de `train(...)`

En cada step:

1. Seleccionar acción con `epsilon_greedy_action(...)`.
2. Hacer `env.step` para obtener `(next_state, reward, terminated, truncated)`.
3. Marcar estado terminal con `done = terminated or truncated`.

### 1.3.4 Bellman update (regla de aprendizaje)

Bloque de update: líneas de TD target/error dentro de `train(...)`

La implementación corresponde a:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

Mapa con el código:

- `best_next = np.max(q_table[next_state])` es $\max_{a'}Q(s',a')$.
- `td_target = reward + gamma * best_next * (1-done)` es el target.
- `td_error = td_target - q_table[state, action]`.
- `q_table[state, action] += alpha * td_error`.

Manejo de terminal state:

- Multiplicar por `(1.0 - float(done))` evita bootstrapping en transiciones terminales.

### 1.3.5 Decay de exploration

Línea de actualización de epsilon dentro de `train(...)`:

$$
\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \cdot \epsilon_{decay})
$$

Esto mueve gradualmente el comportamiento de exploration a exploitation.

### 1.3.6 Monitoreo y evaluación

- Log de entrenamiento cada `log_every` episodios usando `avg100`.
- Evaluación final en `evaluate_policy(...)`:
  - Usa policy puramente greedy (`argmax`).
  - Reporta success rate sobre muchos episodios.

---

## 2) REINFORCE

### 2.1 Qué es REINFORCE

REINFORCE es un algoritmo de **policy gradient**. Optimiza directamente una policy parametrizada $\pi_\theta(a|s)$, en lugar de aprender una Q-table.

Idea central:

- Muestrear trayectorias con la policy actual.
- Aumentar probabilidad de acciones que produjeron alto return.
- Reducir probabilidad de acciones que produjeron bajo return.

### 2.2 Cómo funciona REINFORCE (conceptualmente)

Para un episodio, calcula el return $G_t$ en cada step y actualiza parámetros por gradient ascent del retorno esperado; normalmente se implementa minimizando:

$$
\mathcal{L}(\theta) = -\sum_t \log \pi_\theta(a_t|s_t) G_t
$$

Loop de alto nivel:

1. Ejecutar un episodio con la policy estocástica actual.
2. Calcular discounted returns.
3. Construir la policy-gradient loss con log-probabilities y returns.
4. Hacer backpropagation y actualizar parámetros.

### 2.3 Cómo funciona REINFORCE en este proyecto (mapeo al código)

Archivo principal: `demos/reinforce_cartpole.py`

### 2.3.1 Policy network

Clase: `PolicyNetwork`

- Entrada: vector de observación (`obs_size`).
- Salida: probabilidades de acción con `softmax`.

Interpretación:

- El modelo representa una stochastic policy $\pi_\theta(a|s)$.

### 2.3.2 Cálculo de returns por trayectoria

Función: `discounted_returns(...)`

Para cada time step `t`, el return es:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}
$$

Detalles de implementación:

- Calcula returns hacia atrás.
- Normaliza returns para reducir varianza del gradiente y estabilizar training.

### 2.3.3 Recolección de datos (on-policy)

Loop de episodio: loop de interacción dentro de `train(...)`

En cada step:

1. Convertir observación a tensor.
2. Calcular probabilidades de acción con la policy actual.
3. Hacer sample de acción con `Categorical(probs)`.
4. Guardar `log_prob(action)`.
5. Ejecutar acción y guardar reward.

Por qué el sampling es importante:

- REINFORCE necesita samples de la misma policy que se está optimizando (on-policy Monte Carlo).

### 2.3.4 Policy loss y update

Bloque de loss: acumulación de policy-loss dentro de `train(...)`

Este código implementa:

$$
\mathcal{L}(\theta) = -\sum_t \log \pi_\theta(a_t|s_t) G_t
$$

Interpretación:

- Si $G_t$ es alto, gradient ascent aumenta la probabilidad de la acción muestreada.
- Si $G_t$ es bajo, el update tiende a reducir esa probabilidad.

Paso de optimización:

- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`

### 2.3.5 Señal de convergencia

- Episode reward y moving average (`avg100`) dentro de `train(...)`.
- Condición de “solved” con `avg100 >= solve_score` luego de al menos 100 episodios.

---

## 3) Q-learning vs REINFORCE (resumen para clase)

### Qué se aprende

- Q-learning: aprende **state-action values** `Q(s,a)` y deriva la policy con argmax.
- REINFORCE: aprende la **policy directamente** `\pi_\theta(a|s)`.

### Estilo de update

- Q-learning: temporal-difference bootstrapping (usa `max Q(s',a')`).
- REINFORCE: Monte Carlo policy gradient (usa returns de episodio completo).

### Dónde encaja mejor

- Q-learning (tabular): state spaces pequeños/discretos.
- REINFORCE (neural policy): observaciones continuas o de mayor dimensión.

---

## 4) Hyperparameters en estos scripts

### Q-learning (`parse_args(...)`)

- `alpha`: learning rate.
- `gamma`: discount factor.
- `epsilon-start`, `epsilon-min`, `epsilon-decay`: schedule de exploration.
- `episodes`, `max-steps`: presupuesto de training.

### REINFORCE (`parse_args(...)`)

- `lr`: learning rate del optimizer.
- `gamma`: discount factor.
- `episodes`: máximo de episodios.
- `solve-score`: umbral para early stop.

---

## 5) Tips prácticos para demos

- Si Q-learning se ve plano al inicio, aumenta episodios y/o desacelera `epsilon-decay`.
- Para clase, empieza con FrozenLake determinista (`--slippery` desactivado).
- Para REINFORCE, mantén la normalización de returns y usa suficientes episodios.

---

## 6) Verificación post-training (workflow actualizado)

Los tres demos ahora incluyen modos de verificación post-training:

### 6.1 Render en tiempo real (`human`)

- Q-learning: `--render-eval --render-episodes N`
- REINFORCE: `--render-eval --render-episodes N`
- PPO: `--render-eval --render-episodes N`

Esto abre una ventana interactiva para verificar visualmente si la policy aprendida resuelve la tarea.

### 6.2 Grabación de video (`mp4` para slides)

- Q-learning: `--record-video --video-episodes N --video-dir <path>`
- REINFORCE: `--record-video --video-episodes N --video-dir <path>`
- PPO: `--record-video --video-episodes N --video-dir <path>`

Esto graba episodios de evaluación con Gymnasium `RecordVideo` en:

- `videos/q_learning_frozenlake` (default)
- `videos/reinforce_cartpole` (default)
- `videos/ppo_cartpole` (default)

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

PPO (Proximal Policy Optimization) es un algoritmo moderno de **policy optimization** diseñado para mejorar estabilidad y sample efficiency frente a policy gradient más básicos.

Idea central:

- Optimizar un surrogate objective que desincentiva updates demasiado grandes.
- Reusar datos de trayectorias por varios epochs de optimización.
- Balancear desempeño y estabilidad de training.

En este proyecto, PPO viene de `stable-baselines3`.

### 7.2 Cómo funciona PPO (conceptualmente)

PPO recolecta rollouts, calcula estimaciones de advantage y actualiza policy/value networks con una objective clipping para evitar cambios bruscos de policy en un solo paso.

Loop de alto nivel:

1. Recolectar datos de rollout con la policy actual.
2. Estimar advantages/returns.
3. Ejecutar varios minibatch gradient updates con clipping.
4. Repetir hasta consumir el presupuesto de timesteps.

### 7.3 Cómo funciona PPO en este proyecto (mapeo al código)

Archivo principal: `demos/gymnasium_ppo_cartpole.py`

### 7.3.1 Configuración de training

- `train_and_evaluate` crea `CartPole-v1` con wrapper `Monitor` y construye `PPO` (inicio de función + bloque del constructor).
- El script usa `MlpPolicy` y hyperparameters principales desde CLI: `learning_rate`, `n_steps`, `batch_size`, `gamma` y `seed`.

Interpretación práctica:

- PPO es un algoritmo de policy optimization que mantiene updates estables al limitar cambios bruscos en la policy (esto lo implementa internamente Stable-Baselines3).

### 7.3.2 Learn, save y evaluate

- El training se ejecuta con `model.learn(total_timesteps=args.timesteps)`.
- El guardado del modelo se hace con `model.save(args.model_path)` después de asegurar que exista el directorio padre.
- La evaluación determinista usa `evaluate_policy(..., deterministic=True)` y reporta mean/std reward.

Esto da una métrica clara de “¿aprendió o no?” justo después del training.

### 7.3.3 Verificación visual y grabación

- Ruta de render humano: `render_policy(...)` con `render_mode="human"`, acciones deterministas con `model.predict(..., deterministic=True)`.
- Ruta de grabación en video: `record_policy_video(...)` con `render_mode="rgb_array"` y wrapper `RecordVideo` de Gymnasium.
- Ruta combinada: `--record-and-render` activa render y grabación en una sola ejecución.

### 7.3.4 Parámetros CLI a destacar en clase

- `--timesteps`: presupuesto total de training.
- `--learning-rate`, `--n-steps`, `--batch-size`, `--gamma`: controles centrales de PPO.
- `--eval-episodes`: confiabilidad de la estimación de evaluación.
- `--render-eval`, `--record-video`, `--record-and-render`: modos de verificación de policy post-training.

