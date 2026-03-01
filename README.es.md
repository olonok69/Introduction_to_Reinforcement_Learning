# Demos de Introduction to Reinforcement Learning

Este repositorio contiene tres demos en Python alineados con tu guía de RL:

- **Policy Optimization (REINFORCE)**
- **Q-learning (tabular)**
- **Resolver un environment de Gymnasium con PPO**

El diagrama `rl_algorithms_9_15.svg` se puede usar como apoyo visual mientras ejecutas los scripts.

## Prerrequisitos

- Python 3.10 o 3.11
- `uv` instalado
- Virtual environment existente en `.venv` (ya creado en tu workspace)

## Instalar dependencias

Desde la raíz del proyecto:

```powershell
uv pip install -r requirements.txt
```

## Demos

### 1) REINFORCE en CartPole (Gymnasium)

Script: `demos/reinforce_cartpole.py`

```powershell
python demos/reinforce_cartpole.py
```

Opciones útiles:

```powershell
python demos/reinforce_cartpole.py --episodes 1500 --lr 0.0005 --gamma 0.99
```

Visualizar la policy aprendida después del training:

```powershell
python demos/reinforce_cartpole.py --render-eval --render-episodes 3
```

Grabar episodios de evaluación en video (para slides):

```powershell
python demos/reinforce_cartpole.py --record-video --video-episodes 3 --video-dir videos/reinforce_cartpole
```

Hacer ambas cosas con un solo comando:

```powershell
python demos/reinforce_cartpole.py --record-and-render --render-episodes 3 --video-episodes 3
```

Qué muestra:

- Monte Carlo policy gradient update
- Normalización de returns para entrenamiento más estable
- Promedio móvil de episode reward

### 2) Q-learning tabular en FrozenLake

Script: `demos/q_learning_frozenlake.py`

```powershell
python demos/q_learning_frozenlake.py
```

El mapa determinista (default) entrena más rápido para demo en clase.

Versión estocástica (más difícil, más realista):

```powershell
python demos/q_learning_frozenlake.py --slippery
```

Visualizar la policy aprendida después del training:

```powershell
python demos/q_learning_frozenlake.py --render-eval --render-episodes 5
```

Grabar episodios de evaluación en video (para slides):

```powershell
python demos/q_learning_frozenlake.py --record-video --video-episodes 5 --video-dir videos/q_learning_frozenlake
```

Hacer ambas cosas con un solo comando:

```powershell
python demos/q_learning_frozenlake.py --record-and-render --render-episodes 5 --video-episodes 5
```

Qué muestra:

- Epsilon-greedy exploration
- Bellman update para la Q-table
- Calidad final de la policy mediante evaluación de success rate

### 3) Resolver CartPole de Gymnasium con PPO

Script: `demos/gymnasium_ppo_cartpole.py`

```powershell
python demos/gymnasium_ppo_cartpole.py
```

Este script entrena un agente PPO con `stable-baselines3`, lo guarda en `models/ppo_cartpole` y muestra métricas de evaluación.

Render local opcional después del training:

```powershell
python demos/gymnasium_ppo_cartpole.py --render-demo
```

Render de episodios de evaluación (recomendado):

```powershell
python demos/gymnasium_ppo_cartpole.py --render-eval --render-episodes 3
```

Grabar episodios de evaluación en video (para slides):

```powershell
python demos/gymnasium_ppo_cartpole.py --record-video --video-episodes 3 --video-dir videos/ppo_cartpole
```

Hacer ambas cosas con un solo comando:

```powershell
python demos/gymnasium_ppo_cartpole.py --record-and-render --render-episodes 3 --video-episodes 3
```

## Flujo sugerido para enseñar

1. Empezar con `q_learning_frozenlake.py` (value-based, tabular).
2. Continuar con `reinforce_cartpole.py` (direct policy optimization).
3. Terminar con `gymnasium_ppo_cartpole.py` (algoritmo práctico moderno y uso de librería).

## Documentación adicional

- Explicación detallada (EN): `docs/rl_explanations.md`
- Explicación detallada (ES): `docs/rl_explanations.es.md`

## Cobertura rápida de RL Guide

La documentación del repositorio también se alinea con la teoría central de `RL Guide.pdf`:

- Fundamentos de RL: agente, environment, estado, acción, reward, policy.
- Model-free vs model-based RL (y por qué este repo se enfoca en demos model-free).
- Contraste práctico entre policy optimization y Q-learning.
- Distinción clara entre model, policy y value function.

## Notas

- `requirements.txt` ahora incluye `gymnasium` explícitamente para estos scripts de demo.
- `gym` legacy se mantiene para compatibilidad con código más antiguo durante la modernización.
