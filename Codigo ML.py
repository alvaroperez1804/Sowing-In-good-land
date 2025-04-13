
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


crops = pd.read_csv("soil_measures.csv")

print(crops.isnull().sum())
plt.style.use("dark_background")  # fondo oscuro

sns.set_palette("cool")  # Puedes probar "rocket", "mako", "viridis", etc.

plt.figure(figsize=(10, 6))  # tamaño más cinematográfico
crops["ph"].hist(bins="auto", color="#00FFD1", edgecolor="white", alpha=0.8)

plt.title("🌌 Distribución del pH del Suelo", fontsize=18, color="#00FFB2", weight="bold")
plt.xlabel("🌿 pH", fontsize=14, color="#BBFFFF")
plt.ylabel("📊 Frecuencia", fontsize=14, color="#BBFFFF")

plt.xticks(color='white')
plt.yticks(color='white')

plt.grid(color='gray', linestyle='--', linewidth=0.3, alpha=0.5)

plt.show()
numeric_cols = ["N","P","K","ph"]
matriz_correlacion = crops[numeric_cols].corr()
sns.heatmap(matriz_correlacion, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("matriz de correlacion entre nutrientes y ph")
plt.show()
print(crops.head(101))
print(crops["crop"].value_counts())
datos_totales = crops.count()
print("Numero de datos en total:", datos_totales)
print("numero de veces que aparece el ph", crops["ph"].count())
print("las columnas serian", crops.columns)
#comenzamos el analisis de machine learnings
#definimos cuales son los features y los objetivos, no tomamos los features juntos porque no queremos predecir el conjunto de caracteristicas sino la caracteristica individual que mejor predice el crop
features = ["N","P","K","ph"]
score = {}
best_predictive_feature = {}
best_predictive_feature = {}
target = "crop"
#vemos que variables unicas hay en crop
#print("nombres unicos que crop", crops[target].unique())
#print("las propiedades estadisticas de las columnas features")
#print(crops[features].describe())
promedio_de_elementos_por_cultivo = crops.groupby("crop").mean(numeric_only=True)
print(promedio_de_elementos_por_cultivo)
print(promedio_de_elementos_por_cultivo.describe())

for feature in features:
    X = crops[[feature]]
    y = crops[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #empezamos a entrenar el modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    #hacemos predicciones 
    y_pred = model.predict(X_test)
    #medimos el accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    score[feature] = accuracy

print(score)
valor_maximo = max(score.values())
print(valor_maximo)
indice_del_maximo = max(score, key=score.get)
print(indice_del_maximo)
best_predictive_feature = {indice_del_maximo:valor_maximo}
print(best_predictive_feature)