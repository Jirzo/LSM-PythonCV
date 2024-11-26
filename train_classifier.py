import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar datos
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspeccionar los datos
print(f"Tipo de data: {type(data_dict['data'])}")
print(f"Longitudes de las secuencias: {[len(item) for item in data_dict['data'][:5]]}")

# Normalizar las secuencias (si es necesario)
max_len = max(len(item) for item in data_dict['data'])
data = np.array([item + [0] * (max_len - len(item)) for item in data_dict['data']])

# Convertir etiquetas a un arreglo de NumPy
labels = np.asarray(data_dict['labels'])

# Verificar que las dimensiones coincidan
assert len(data) == len(labels), "El número de datos y etiquetas no coincide."

# Dividir los datos
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Entrenar el modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predecir y evaluar
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f"Precisión del modelo: {accuracy}")

score = accuracy_score(y_predict, y_test)
print('{}% of sambles were classified correctlu!'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'mocel': model}, f)
f.close()