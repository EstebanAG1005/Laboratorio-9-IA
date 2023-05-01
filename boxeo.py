import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np

def get_q_values(observation, model):
    # Agregar una dimensión adicional a la observación para que sea compatible con la entrada del modelo
    input_state = tf.expand_dims(observation, axis=0)

    # Obtener los Q-values para todas las acciones posibles
    q_values = model.predict(input_state)

    return q_values[0]

env = gym.make('Boxing-v0', render_mode='human')
# Reiniciar el entorno y obtener la observación inicial
obs = env.reset()

# Crear el modelo de red neuronal
model = keras.Sequential([
    keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(210, 160, 3)),
    keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(env.action_space.n)
])
model.compile(optimizer=keras.optimizers.Adam(lr=0.00025), loss='mse')

# Hiperparámetros del agente
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.99
gamma = 0.99
batch_size = 32

# Lista para almacenar los pasos de cada episodio
episode_steps = []

# Bucle principal para iterar sobre el entorno
while True:
    # Renderizar el entorno en modo 'human'
    env.render()

    # Tomar una acción con un valor aleatorio de 'epsilon-greedy'
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        q_values = get_q_values(obs, model)
        action = tf.argmax(q_values).numpy()

    # Ejecutar la acción y obtener la observación, recompensa, estado "done" e información
    next_obs, reward, done, truncate, info = env.step(action)

    # Almacenar el paso en la lista de pasos del episodio
    episode_steps.append((obs, action, reward, next_obs, done))

    # Actualizar la observación actual
    obs = next_obs

    # Verificar si el episodio ha terminado y actualizar el modelo
    if done:
        # Imprimir el número de pasos del episodio
        print(f"Episode finished after {len(episode_steps)} steps")

        # Actualizar el valor de 'epsilon'
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Crear los arrays para las entradas y las salidas del modelo
        X = np.zeros((len(episode_steps), 210, 160, 3))
        y = np.zeros((len(episode_steps), env.action_space.n))

        # Iterar sobre los pasos del episodio en orden inverso
        for i in range(len(episode_steps) - 1, -1, -1):
            obs, action, reward, next_obs, done = episode_steps[i]

            if type(obs) != tuple:

                # Calcular el nuevo valor 'target' para el Q-value
                if done:
                    target = reward
                else:
                    q_values = get_q_values(next_obs, model)
                    target = reward + gamma * np.max(q_values)

                # Actualizar los arrays de entradas y salidas
                print(obs.shape)
                X[i] = obs
                y[i] = get_q_values(obs, model)
                y[i][action] = target

        # Entrenar el modelo con los datos recolectados
        model.train_on_batch(X, y)

        # Reiniciar la lista de pasos del episodio
        episode_steps = []

        # Reiniciar el entorno y obtener la observación inicial
        obs = env.reset()

