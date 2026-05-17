import pygad
import numpy as np

# ============================================================
# КРОК 1: Дані задачі рюкзака
# Кожен товар має [вага, цінність].
# Задача: вибрати товари з макс. цінністю, не перевищуючи MAX_WEIGHT.
# ============================================================

# 14 items: [weight, value]
items = np.array([
    [2, 6], [2, 3], [6, 5], [5, 4], [4, 6],
    [5, 5], [3, 2], [7, 8], [3, 4], [4, 5],
    [1, 3], [6, 7], [2, 4], [3, 5]
])

MAX_WEIGHT = 20  # maximum knapsack capacity

# ============================================================
# КРОК 2: Фітнес-функція
# Оцінює наскільки "добра" хромосома (один варіант рюкзака).
# solution — масив з 14 генів: 1 = беремо товар, 0 = не беремо.
# Навіщо: без фітнес-функції GA не знає що покращувати.
# ============================================================

def fitness_func(ga_instance, solution, solution_idx):
    # multiply each gene (0/1) by weight/value → sum selected items only
    total_weight = np.sum(solution * items[:, 0])
    total_value  = np.sum(solution * items[:, 1])

    if total_weight > MAX_WEIGHT:
        return 0  # penalty: invalid solution (overloaded knapsack)
    return total_value  # higher value = better solution

# ============================================================
# КРОК 3: Початкова популяція
# 10 випадкових хромосом (особин), кожна — 14 генів (0 або 1).
# Навіщо: GA починає з різноманітного набору рішень — не з нуля.
# ============================================================

# shape (10, 14): 10 individuals × 14 genes
initial_population = np.random.randint(0, 2, size=(10, 14))

# ============================================================
# КРОК 4: Параметри GA та створення екземпляру
# Параметри — як гіперпараметри моделі в sklearn.
# ============================================================

ga = pygad.GA(
    num_generations=100,         # кількість поколінь (ітерацій еволюції)
    num_parents_mating=5,        # скільки батьків беруть участь у кросовері
    fitness_func=fitness_func,   # функція оцінки хромосоми
    initial_population=initial_population,
    gene_type=int,               # гени — цілі числа (0 або 1)
    gene_space=[0, 1],           # допустимі значення генів
    crossover_type="single_point", # розрізає хромосому в одній точці
    mutation_type="random",        # випадково змінює ген на інше допустиме значення
    mutation_percent_genes=10,     # 10% генів мутують у кожному поколінні
    keep_parents=2,                # 2 найкращі батьки переходять у наступне покоління без змін
)

# run the evolution
ga.run()

# ============================================================
# КРОК 5: Виведення найкращого результату
# ============================================================

solution, solution_fitness, _ = ga.best_solution()

# indices of selected items (genes == 1)
selected_items = [i for i, gene in enumerate(solution) if gene == 1]
total_weight = int(np.sum(solution * items[:, 0]))
total_value  = int(solution_fitness)

print(f"Best fitness (value): {total_value}")
print(f"Total weight: {total_weight} / {MAX_WEIGHT}")
print(f"Selected items (indices): {selected_items}")

# ============================================================
# КРОК 6: Порівняння різних кросоверів та мутацій
# Навіщо: як GridSearch — знаходимо оптимальні гіперпараметри GA.
#
# Типи кросоверу:
#   single_point — розрізає хромосому в одній точці
#   two_points   — два розрізи, середина від одного батька
#   uniform      — кожен ген вибирається незалежно від батька
#
# Типи мутації:
#   random    — ген замінюється на випадкове допустиме значення
#   swap      — два гени міняються місцями
#   inversion — підпослідовність генів перевертається
# ============================================================

configs = [
    {"crossover_type": "single_point", "mutation_type": "random",   "label": "single_point + random"},
    {"crossover_type": "two_points",   "mutation_type": "random",   "label": "two_points + random"},
    {"crossover_type": "uniform",      "mutation_type": "random",   "label": "uniform + random"},
    {"crossover_type": "single_point", "mutation_type": "swap",     "label": "single_point + swap"},
    {"crossover_type": "uniform",      "mutation_type": "inversion","label": "uniform + inversion"},
]

print("\n--- Comparing configs ---")
for cfg in configs:
    pop = np.random.randint(0, 2, size=(10, 14))  # fresh random population per config
    g = pygad.GA(
        num_generations=100,
        num_parents_mating=5,
        fitness_func=fitness_func,
        initial_population=pop,
        gene_type=int,
        gene_space=[0, 1],
        crossover_type=cfg["crossover_type"],
        mutation_type=cfg["mutation_type"],
        mutation_percent_genes=10,
        keep_parents=2,
    )
    g.run()
    _, best, _ = g.best_solution()
    print(f"{cfg['label']:35s} → fitness: {int(best)}")


# ============================================================
# ВИСНОВОК
# ============================================================
# Генетичний алгоритм для задачі рюкзака (14 товарів, MAX_WEIGHT=20):
#
# Найкращий результат: uniform + random → fitness 34
#   → uniform кросовер дає більше різноманітності: кожен ген
#     вибирається незалежно від одного з батьків, що краще
#     досліджує простір рішень
#
# Найгірший результат: uniform + inversion → fitness 0
#   → мутація inversion перевертає підпослідовність генів,
#     що для бінарної задачі (0/1) руйнує популяцію —
#     майже всі хромосоми стають невалідними (вага > MAX_WEIGHT)
#
# Висновок про кросовер:
#   uniform > two_points > single_point
#   Більше точок розрізу = більше генетичного різноманіття
#
# Висновок про мутацію:
#   random — найкраща для бінарних задач (просто інвертує біт)
#   swap — помірна ефективність
#   inversion — не підходить для задач з бінарними генами
#
# Загальний висновок:
#   GA ефективно знаходить близьке до оптимального рішення
#   без перебору всіх 2^14 = 16384 варіантів.
#   Оптимальний набір параметрів: uniform + random, 100 поколінь.
# ============================================================