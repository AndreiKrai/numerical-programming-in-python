# =============================================================
# Policy Iteration — навчання з підкріпленням на FrozenLake
# Ланцюги Маркова + оптимальна стратегія (policy)
# =============================================================

# --- Імпорти ---
import numpy as np       # математика: zeros, argmax, reshape
import gymnasium as gym  # середовище FrozenLake (OpenAI Gym)

# =============================================================
# КРОК 1: Завантажити середовище FrozenLake-v1
# =============================================================
# FrozenLake: сітка 4×4, агент іде з S (старт) до G (ціль)
# Уникаючи H (дірки). Лід слизький — дії стохастичні.
#
# Стани: 0-15 (кожна клітинка сітки)
# Дії:   0=← 1=↓ 2=→ 3=↑
# Нагорода: +1 якщо досяг G, 0 інакше
#
# is_slippery=True — рух не детермінований (ланцюг Маркова)

env = gym.make('FrozenLake-v1', is_slippery=True)

print("Кількість станів:", env.observation_space.n)   # 16
print("Кількість дій:", env.action_space.n)            # 4

# env.unwrapped.P[state][action] — таблиця переходів (ланцюг Маркова)
# Повертає список: [(prob, next_state, reward, done), ...]
print("Таблиця переходів стану 0, дія 0:")
print(env.unwrapped.P[0][0])

# =============================================================
# КРОК 2: compute_value_function()
# =============================================================
# Оцінює "цінність" кожного стану при поточній політиці.
# Формула Беллмана: V(s) = Σ P(s'|s,a) * [R + γ * V(s')]
#
# Параметри:
#   policy — поточна стратегія (масив дій для кожного стану)
#   gamma  — дисконт (0.99): майбутнє трохи менш цінне ніж зараз
#   theta  — поріг збіжності: зупиняємось коли зміни малі

def compute_value_function(policy, env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.observation_space.n)  # початкові цінності = 0

    while True:
        delta = 0  # максимальна зміна V за цю ітерацію

        for state in range(env.observation_space.n):
            v = 0
            action = policy[state]  # дія згідно поточної політики

            # Сума по всіх можливих переходах (ланцюг Маркова):
            # prob — ймовірність переходу (лід слизький → < 1.0)
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                v += prob * (reward + gamma * V[next_state])

            delta = max(delta, abs(v - V[state]))  # відстежуємо зміну
            V[state] = v

        if delta < theta:  # збіглись → зупиняємось
            break

    return V  # вектор цінностей (16,) для кожного стану

# =============================================================
# КРОК 3: policy_iteration()
# =============================================================
# Ітеративно покращує політику до оптимальної.
# Два кроки що чергуються:
#   1. Policy Evaluation: оцінити V для поточної policy
#   2. Policy Improvement: оновити policy → обрати кращу дію в кожному стані

def policy_iteration(env, gamma=0.99):
    # Початкова політика — дія 0 (←) для всіх станів
    policy = np.zeros(env.observation_space.n, dtype=int)

    while True:
        # Крок 1: оцінити поточну політику
        V = compute_value_function(policy, env, gamma)

        policy_stable = True  # припускаємо що більше не зміниться

        for state in range(env.observation_space.n):
            old_action = policy[state]

            # Крок 2: для кожного стану знайти кращу дію
            # Перебрати всі 4 дії і порахувати їх очікувану цінність
            action_values = []
            for action in range(env.action_space.n):
                value = 0
                for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                    value += prob * (reward + gamma * V[next_state])
                action_values.append(value)

            # Обрати дію з максимальною очікуваною цінністю
            policy[state] = np.argmax(action_values)

            # Якщо дія змінилась — потрібна ще одна ітерація
            if old_action != policy[state]:
                policy_stable = False

        if policy_stable:  # жодна дія не змінилась → оптимальна політика!
            break

    return policy, V

# =============================================================
# КРОК 4: Візуалізація оптимальної політики
# =============================================================
# show_render() малює сітку 4×4 зі стрілками — куди іти в кожній клітинці

def show_render(policy):
    actions = {0: '←', 1: '↓', 2: '→', 3: '↑'}  # mapping дій → символи

    print("\nОптимальна політика (4×4):")
    print("-" * 13)
    for row in range(4):
        line = "| "
        for col in range(4):
            state = row * 4 + col          # номер стану = рядок*4 + колонка
            line += actions[policy[state]] + " | "
        print(line)
        print("-" * 13)

# --- Запуск ---
policy, V = policy_iteration(env)  # знайти оптимальну політику

show_render(policy)  # вивести стрілки на сітку

# Вивести цінності станів у вигляді 4×4 матриці
# Клітинки біля G мають найвищі значення
print("\nЦінності станів V(s) — 4×4:")
print(V.reshape(4, 4).round(3))

print("""
=== ВИСНОВОК ===

Policy Iteration знайшла оптимальну стратегію для FrozenLake.

Алгоритм використовує ланцюги Маркова:
- env.P[s][a] — таблиця ймовірностей переходів (Марковська властивість)
- Майбутній стан залежить тільки від поточного, не від минулого

Два кроки що чергуються:
1. compute_value_function → оцінює наскільки добрий кожен стан
2. policy_iteration → покращує стратегію поки вона не стане оптимальною

Стани біля Goal (G) мають найвищі цінності V(s).
Стрілки показують куди іти агенту в кожній клітинці.
""")

print("""
┌─────────────────────────────────────────────────────┐
│           СХЕМА ПОТОКУ POLICY ITERATION             │
└─────────────────────────────────────────────────────┘

  FrozenLake середовище (4×4 = 16 станів, 4 дії)
  env.P[state][action] → таблиця переходів (Марков)
                │
                ▼
  Початкова policy = [0, 0, 0, ..., 0]  (всі ← )
                │
         ┌──────▼──────┐
         │  Ітерація   │
         └──────┬──────┘
                │
                ▼
  compute_value_function(policy)
  V(s) = Σ P(s'|s,a) * [R + γ*V(s')]  ← формула Беллмана
                │
                ▼
  Policy Improvement:
  для кожного стану → знайти дію з max Σ P*[R + γ*V]
                │
         ┌──────▼──────────────┐
         │ policy змінилась?   │
         │  Так → ще ітерація  │
         │  Ні  → ОПТИМАЛЬНА! │
         └─────────────────────┘
                │
                ▼
         show_render(policy)
         ← ↓ → ↑ ← ↓ → ↑ ...  (стрілки на сітці)
""")
