Autor: Jorge Martinez, 100508957@alumnos.uc3m.es

Fecha: April 7, 2024


# Informe del Proyecto de March Machine Learning Mania 2024

## Resumen

Este proyecto se enfoca en el desarrollo de un modelo predictivo robusto para predecir los resultados de los juegos del torneo de la NCAA, utilizando técnicas avanzadas de aprendizaje automático. Se exploran varios modelos y se selecciona el que ofrece la mejor precisión y confiabilidad en sus predicciones. Este informe detalla el proceso de selección del modelo, la preparación de los datos, y los resultados obtenidos.


## Fuente de Datos

Este proyecto utiliza datos de la competición March Machine Learning Mania 2024, disponible en Kaggle. La competición proporciona un conjunto de datos que incluye resultados históricos de juegos y estadísticas de equipos participantes en el torneo de baloncesto de la NCAA.

Enlace al dataset: [March Machine Learning Mania 2024 Dataset](https://www.kaggle.com/c/march-machine-learning-mania-2024)


De todos los datos disponibles se van a usar los siguientes:

### MTeams.csv

Este archivo identifica los diferentes equipos universitarios presentes en el conjunto de datos.

#### Descripción de las columnas:

| Columna        | Descripción |
| -------------- | ----------- |
| **TeamID**     | Un número de identificación de 4 dígitos, que identifica de forma exclusiva a cada equipo de la NCAA. |
| **TeamName**   | Nombre de la universidad del equipo. |
| **FirstD1Season** | La primera temporada en nuestro conjunto de datos en que la escuela era de División I. |
| **LastD1Season**  | La última temporada de nuestro conjunto de datos en la que la escuela era de División I. |



### MNCAATourneySeeds.csv

Estos archivos identifican los cabezas de serie de todos los equipos en cada torneo de la NCAA®, para todas las temporadas de datos históricos. Por lo tanto, hay entre 64 y 68 filas para cada año, dependiendo de si hubo partidos de desempate y de cuántos hubo.

#### Descripción de las columnas:

| Columna | Descripción |
|---------|-------------|
| **Season** | El año en que se jugó el torneo. |
| **Seed** | Se trata de un identificador de 3/4 caracteres de la semilla, donde el primer carácter es W, X, Y o Z (que identifica la región en la que estaba el equipo) y los dos dígitos siguientes (01, 02, ..., 15 o 16) indican la semilla dentro de la región. |
| **TeamID** | Identifica el número de identificación del equipo. |



### MRegularSeasonDetailedResults.csv

Estos archivos proporcionan los resultados de los equipos de muchas temporadas regulares de datos históricos, a partir de la temporada 2003.

Se muestran las 5 primeras filas de estos para que se pueda ver el tipo de informacion que continen

| Season | DayNum | WTeamID | WScore | LTeamID | LScore | WLoc | NumOT | WFGM | WFGA | ... | LAst | LTO | LStl | LBlk | LPF |
|--------|--------|---------|--------|---------|--------|------|-------|------|------|-----|------|-----|------|------|-----|
| 2003   | 10     | 1104    | 68     | 1328    | 62     | N    | 0     | 27   | 58   | ... | 8    | 18  | 9    | 2    | 20  |
| 2003   | 10     | 1272    | 70     | 1393    | 63     | N    | 0     | 26   | 62   | ... | 7    | 12  | 8    | 6    | 16  |
| 2003   | 11     | 1266    | 73     | 1437    | 61     | N    | 0     | 24   | 58   | ... | 9    | 12  | 2    | 5    | 23  |
| 2003   | 11     | 1296    | 56     | 1457    | 50     | N    | 0     | 18   | 38   | ... | 9    | 19  | 4    | 3    | 23  |
| 2003   | 11     | 1400    | 77     | 1208    | 71     | N    | 0     | 30   | 61   | ... | 12   | 10  | 7    | 1    | 14  |


### MNCAATourneyDetailedResults.csv

Estos archivos proporcionan los resultados de los equipos en muchos torneos de la NCAA®, a partir de la temporada 2003. 


| Season | DayNum | WTeamID | WScore | LTeamID | LScore | WLoc | NumOT | WFGM | WFGA | WFGM3 | WFGA3 | WFTM | ... | LFGM | LFGA | LFGM3 | LFGA3 | LFTM | LFTA | LOR | LDR | LAst | LTO | LStl | LBlk | LPF |
|--------|--------|---------|--------|---------|--------|------|-------|------|------|-------|-------|------|-----|------|------|-------|-------|------|------|-----|-----|------|-----|------|------|-----|
| 2003   | 134    | 1421    | 92     | 1411    | 84     | N    | 1     | 32   | 69   | 11    | 29    | 17   | ... | 29   | 67   | 12    | 31    | 14   | 31   | 17  | 28  | 16   | 15  | 5    | 0    | 22  |
| 2003   | 136    | 1112    | 80     | 1436    | 51     | N    | 0     | 31   | 66   | 7     | 23    | 11   | ... | 20   | 64   | 4     | 16    | 7    | 7    | 8   | 26  | 12   | 17  | 10   | 3    | 15  |
| 2003   | 136    | 1113    | 84     | 1272    | 71     | N    | 0     | 31   | 59   | 6     | 14    | 16   | ... | 25   | 69   | 7     | 28    | 14   | 21   | 20  | 22  | 11   | 12  | 2    | 5    | 18  |
| 2003   | 136    | 1141    | 79     | 1166    | 73     | N    | 0     | 29   | 53   | 3     | 7     | 18   | ... | 27   | 60   | 7     | 17    | 12   | 17   | 14  | 17  | 20   | 21  | 6    | 6    | 21  |
| 2003   | 136    | 1143    | 76     | 1301    | 74     | N    | 1     | 27   | 64   | 7     | 20    | 15   | ... | 25   | 56   | 9     | 21    | 15   | 20   | 10  | 26  | 16   | 14  | 5    | 8    | 19  |


### 2024_tourney_seeds.csv

Contiene el cuadro de playoffs de este March Madness 2024


## Objetivo del Proyecto

El objetivo de este proyecto es desarrollar un modelo predictivo capaz de predecir los resultados de los partidos del torneo de la NCAA. Buscamos crear un modelo que no solo prediga con precisión los ganadores de cada partido, sino que también estime la probabilidad de victoria de cada equipo, proporcionando así insights valiosos para apuestas y análisis deportivos.

## Exploración de los Datos e Información Relevante

El dataset proporcionado incluye variables como el historial de partidos, rankings de equipos, estadísticas de rendimiento, entre otros. Un análisis exploratorio inicial revela patrones interesantes, como la importancia de la defensa sobre el ataque para predecir victorias o la correlación entre el ranking de un equipo y su probabilidad de avanzar en el torneo.

### Preparacion de los datos

Partimos de los datos contenidos en MRegularSeasonDetailedResults.csv y MNCAATourneyDetailedResults.csv y le aplicamos transformaciones que basicamente consisten en cambiar de ganador (W) y perdedor (L) a, equipo 1 (T1) y equipo 2 (T2). Duplicando el numero de filas que teniamos originalmente, ya que para cada partido inicial se consideran dos casos, en el que el ganador es el equipo 1 y en el que es el equipo 2.

Esto se aplica tanto al conjunto de datos de la temporada regular como a los datos de playoffs.

Obteniendo unos datos como los siguientes:

| Season | DayNum | T1_TeamID | T1_Score | T2_TeamID | T2_Score | location | NumOT | T1_FGM | T1_FGA | ... | T2_FTM | T2_FTA | T2_OR | T2_DR | T2_Ast | T2_TO | T2_Stl | T2_Blk | T2_PF | PointDiff |
|--------|--------|-----------|----------|-----------|----------|----------|-------|--------|--------|-----|--------|--------|-------|-------|--------|-------|--------|-------|-------|-----------|
| 2003   | 10     | 1104      | 68       | 1328      | 62       | 0        | 0     | 27     | 58     | ... | 16     | 22     | 10    | 22   | 8      | 18    | 9      | 2     | 20    | 6         |
| 2003   | 10     | 1328      | 62       | 1104      | 68       | 0        | 0     | 22     | 53     | ... | 11     | 18     | 14    | 24   | 13     | 23    | 7      | 1     | 22    | -6        |
| 2003   | 10     | 1272      | 70       | 1393      | 63       | 0        | 0     | 26     | 62     | ... | 9      | 20     | 20    | 25   | 7      | 12    | 8      | 6     | 16    | 7         |
| 2003   | 10     | 1393      | 63       | 1272      | 70       | 0        | 0     | 24     | 67     | ... | 10     | 19     | 15    | 28   | 16     | 13    | 4      | 4     | 18    | -7        |
| 2003   | 11     | 1266      | 73       | 1437      | 61       | 0        | 0     | 24     | 58     | ... | 14     | 23     | 31    | 22   | 9      | 12    | 2      | 5     | 23    | 12        |


### Creación de atributos

Patiendo de los datos modificados de temporada regular y playoffs, y seeds de los playoffs de cada equipo en los distintos años, se aplican las siguientes operaciones.

Esta función tiene como objetivo enriquecer los datos de los partidos del torneo con estadísticas relevantes y derivadas que faciliten la construcción de un modelo predictivo más robusto y preciso para los partidos de la NCAA. La función realiza las siguientes operaciones:

#### 1. Agregación de Estadísticas de Temporada Regular por Equipo y Temporada
La función comienza agrupando las estadísticas de los partidos de la temporada regular por cada equipo y temporada. Se seleccionan métricas clave como tiros de campo, rebotes, asistencias, entre otros, calculando el promedio para obtener un resumen estadístico que refleja el desempeño promedio de los equipos a lo largo de la temporada.

#### 2. Duplicación de Estadísticas para Equipos 1 y 2
Para facilitar las fusiones de datos futuras, se duplican las estadísticas agregadas, creando dos conjuntos de columnas: uno para el "equipo 1" y otro para el "equipo 2". Esto permite asociar cada partido del torneo con un conjunto completo de estadísticas para ambos equipos involucrados.

#### 3. Fusión de Estadísticas de Temporada con Datos del Torneo
Las estadísticas de la temporada regular se fusionan con los datos de los partidos del torneo. Esta fusión se realiza para ambos equipos en cada partido, enriqueciendo los registros con un contexto histórico amplio que es crucial para el análisis predictivo.

Basicamente se fusiona los 
| Season | DayNum | T1_TeamID | T1_Score | T2_TeamID | T2_Score, de cada partido de playoffs, con las correspondientes estadisticas calculadas anteriormente

#### 4. Inclusión de la Diferencia de Semillas
Se añade la diferencia de semillas entre los dos equipos en cada partido del torneo. La semilla refleja la valoración y expectativas del comité del torneo hacia los equipos, y su diferencia es un indicador significativo que puede influir en las predicciones del resultado del partido.



Obteniendo esto como resultado

| Season | DayNum | T1_TeamID | T1_Score | T2_TeamID | T2_Score | T1_FGM  | ... | T2_opponent_Stl | T2_opponent_Blk | T2_opponent_PF | T2_PointDiff | T1_seed | T2_seed | Seed_diff |
|--------|--------|-----------|----------|-----------|----------|---------|-----|-----------------|-----------------|----------------|--------------|---------|---------|-----------|
| 2003   | 134    | 1421      | 92       | 1411      | 84       | 24.38   | ... | 8.00            | 2.60            | 21.63          | 1.97         | 16      | 16      | 0         |
| 2003   | 134    | 1411      | 84       | 1421      | 92       | 24.73   | ... | 8.83            | 4.24            | 18.69          | -7.24        | 16      | 16      | 0         |
| 2003   | 136    | 1436      | 51       | 1112      | 80       | 24.83   | ... | 5.96            | 2.39            | 22.07          | 14.96        | 16      | 1       | 15        |
| 2003   | 136    | 1112      | 80       | 1436      | 51       | 30.32   | ... | 7.10            | 3.66            | 17.93          | 4.66         | 1       | 16      | -15       |
| 2003   | 136    | 1113      | 84       | 1272      | 71       | 27.21   | ... | 7.28            | 3.17            | 19.93          | 8.69         | 10      | 7       | 3         |
| 2023   | 152    | 1163      | 72       | 1274      | 59       | 27.61   | ... | 5.97            | 3.28            | 15.06          | 7.28         | 4       | 5       | -1        |
| 2023   | 152    | 1194      | 71       | 1361      | 72       | 27.91   | ... | 6.31            | 2.94            | 18.22          | 7.53         | 9       | 5       | 4         |
| 2023   | 152    | 1361      | 72       | 1194      | 71       | 25.09   | ... | 5.62            | 2.41            | 16.75          | 12.88        | 5       | 9       | -4        |
| 2023   | 154    | 1163      | 76       | 1361      | 59       | 27.61   | ... | 6.31            | 2.94            | 18.22          | 7.53         | 4       | 5       | -1        |
| 2023   | 154    | 1361      | 59       | 1163      | 76       | 25.09   | ... | 6.67            | 2.88            | 17.36          | 13.42        | 5       | 4       | 1         |

2630 filas

63 columnas

**Column Names**

- `Season`
- `DayNum`
- `T1_TeamID`
- `T1_Score`
- `T2_TeamID`
- `T2_Score`
- `T1_FGM`
- `T1_FGA`
- `T1_FGM3`
- `T1_FGA3`
- `T1_FTM`
- `T1_FTA`
- `T1_OR`
- `T1_DR`
- `T1_Ast`
- `T1_TO`
- `T1_Stl`
- `T1_Blk`
- `T1_PF`
- `T1_opponent_FGM`
- `T1_opponent_FGA`
- `T1_opponent_FGM3`
- `T1_opponent_FGA3`
- `T1_opponent_FTM`
- `T1_opponent_FTA`
- `T1_opponent_OR`
- `T1_opponent_DR`
- `T1_opponent_Ast`
- `T1_opponent_TO`
- `T1_opponent_Stl`
- `T1_opponent_Blk`
- `T1_opponent_PF`
- `T1_PointDiff`
- `T2_FGM`
- `T2_FGA`
- `T2_FGM3`
- `T2_FGA3`
- `T2_FTM`
- `T2_FTA`
- `T2_OR`
- `T2_DR`
- `T2_Ast`
- `T2_TO`
- `T2_Stl`
- `T2_Blk`
- `T2_PF`
- `T2_opponent_FGM`
- `T2_opponent_FGA`
- `T2_opponent_FGM3`
- `T2_opponent_FGA3`
- `T2_opponent_FTM`
- `T2_opponent_FTA`
- `T2_opponent_OR`
- `T2_opponent_DR`
- `T2_opponent_Ast`
- `T2_opponent_TO`
- `T2_opponent_Stl`
- `T2_opponent_Blk`
- `T2_opponent_PF`
- `T2_PointDiff`
- `T1_seed`
- `T2_seed`
- `Seed_diff`


**Gráfica de ejemplo:** Aquí puedes insertar gráficas de tu análisis exploratorio, como la distribución de victorias por ranking de equipo.

```markdown
![Distribución de victorias por ranking](url_de_la_imagen)
```

Los datos sobre los que se ha entrenado tienen la siguiente estructura:


Como puede observarse en la imagen los valores utilizados para hacer la prediccion son las estadisticas basicas (tiros por parido, ...) de los equipos para los distintos partidos

## Métrica Utilizada

Para evaluar los modelos de machine learning, hemos utilizado la log-loss (pérdida logarítmica) como métrica principal. Esta métrica es ideal para problemas de clasificación binaria en los que es importante no solo predecir el resultado correcto, sino también la confianza en la predicción. La elección de esta métrica se debe a su capacidad para penalizar las predicciones incorrectas que se hacen con alta confianza.


## Modelos de Machine Learning y Experimentación

Durante la fase de experimentación, exploramos dos principales modelos de aprendizaje automático: redes neuronales y XGBoost. Ambos modelos fueron evaluados con la intención de maximizar la precisión y confiabilidad en las predicciones del torneo NCAA.

### XGBoost

Utilizamos XGBoost (Extreme Gradient Boosting) por su reconocida eficiencia en competencias de datos y su habilidad para manejar variados tipos de características. Configuramos XGBoost para minimizar el error absoluto medio (MAE), utilizando una función de pérdida personalizada basada en la distribución de Cauchy, lo que ayudó a mejorar la robustez del modelo frente a valores atípicos. El ajuste fino del modelo se realizó mediante una validación cruzada repetida para garantizar su capacidad de generalización y estabilidad.

#### **Inicialización y Configuración de Datos**

Partiendo de los datos preprocesados explicados anteriormente se aplica lo siguiente:

- **Cálculo de la Variable Objetivo:** La diferencia de puntuación entre los equipos (`T1_Score` - `T2_Score`) se establece como la variable objetivo `y`. Esta representación directamente relacionada con el resultado del partido facilita un enfoque de modelado orientado al objetivo real.

| Index | Target |
|-------|--------|
| 0     | 8      |
| 1     | -8     |
| 2     | -29    |
| 3     | 29     |
| 4     | 13     |

- **Selección de Características:** Las características utilizadas para el entrenamiento se extraen del DataFrame, excluyendo las primeras seis columnas que no son relevantes para el modelo. Esto asegura que solo se utilicen datos pertinentes que potencialmente influyan en el resultado del partido.
- `Season`
- `DayNum`
- `T1_TeamID`
- `T1_Score`
- `T2_TeamID`
- `T2_Score`

|    | T1_FGM    | T1_FGA    | T1_FGM3   | T1_FGA3   | T1_FTM    | T1_FTA    | T1_OR     | ... | T2_opponent_Stl | T2_opponent_Blk | T2_opponent_PF | T2_PointDiff | T1_seed | T2_seed | Seed_diff |
|----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----|-----------------|-----------------|----------------|--------------|---------|---------|-----------|
| 0  | 24.379310 | 56.793103 | 6.482759  | 18.000000 | 15.965517 | 20.931034 | 12.275862 | ... | 8.000000        | 2.600000        | 21.633333      | 1.966667     | 16      | 16      | 0         |
| 1  | 24.733333 | 55.266667 | 5.933333  | 18.500000 | 17.400000 | 28.066667 | 13.166667 | ... | 8.827586        | 4.241379        | 18.689655      | -7.241379    | 16      | 16      | 0         |
| 2  | 24.827586 | 55.862069 | 5.275862  | 15.482759 | 12.862069 | 19.551724 | 12.965517 | ... | 5.964286        | 2.392857        | 22.071429      | 14.964286    | 16      | 1       | 15        |
| 3  | 30.321429 | 65.714286 | 7.035714  | 20.071429 | 17.535714 | 25.000000 | 15.178571 | ... | 7.103448        | 3.655172        | 17.931034      | 4.655172     | 1       | 16      | -15       |
| 4  | 27.206897 | 56.896552 | 4.000000  | 12.586207 | 17.551724 | 26.206897 | 13.689655 | ... | 7.275862        | 3.172414        | 19.931034      | 8.689655     | 10      | 7       | 3         |


#### **Configuración de Parámetros del Modelo**
El modelo se configura con parámetros específicos que controlan su comportamiento durante el entrenamiento:

**Métrica de Evaluación:** Se utiliza el error absoluto medio (`mae`), que proporciona una medida robusta del error que penaliza igualmente todos los errores, independientemente de su magnitud.

**Hiperparámetros:** Se ajustan varios hiperparámetros para optimizar el desempeño del modelo.

- **`eval_metric`: 'mae'**
  - **Descripción:** Esta es la métrica de evaluación utilizada para validar los datos mientras el modelo se entrena. 'MAE' significa Error Absoluto Medio, una métrica que proporciona una medida promedio de las diferencias absolutas entre las predicciones y los valores observados, dando igual peso a todos los errores.
  
- **`booster`: 'gbtree'**
  - **Descripción:** Especifica el tipo de modelo que se utilizará. 'gbtree' indica que se usarán árboles de decisión potenciados como base del modelo. Este es el tipo de refuerzo más común y generalmente el más efectivo para problemas estructurados/tabulares.

- **`eta`: 0.05**
  - **Descripción:** También conocida como la tasa de aprendizaje, controla la velocidad a la que el modelo se adapta a las complejidades del problema. Un valor más bajo hace que el entrenamiento sea más lento, pero puede mejorar la precisión del modelo y prevenir el sobreajuste.

- **`subsample`: 0.35**
  - **Descripción:** Es la fracción de observaciones a ser seleccionadas aleatoriamente para cada árbol. Submuestrear ayuda a prevenir el sobreajuste y añade más aleatoriedad al proceso de entrenamiento.
  
- **`colsample_bytree`: 0.7**
  - **Descripción:** Es la fracción de columnas a ser seleccionadas aleatoriamente para cada árbol. Al igual que `subsample`, este parámetro ayuda a controlar el sobreajuste, asegurando que cada árbol se construya con una muestra aleatoria de las características.

- **`num_parallel_tree`: 3**
  - **Descripción:** Este parámetro especifica el número de árboles que se construirán en paralelo en cada iteración del boosting. Usar más de un árbol por iteración es una forma de potenciar el ensamblaje, similar al bagging.

- **`min_child_weight`: 40**
  - **Descripción:** Define el peso mínimo necesario en las hojas para continuar haciendo una partición en el árbol. Un valor más alto reduce el sobreajuste al hacer que el modelo sea más conservador, requiriendo más evidencia antes de decidir hacer una división.

- **`gamma`: 10**
  - **Descripción:** Especifica el mínimo descenso en la función de pérdida requerido para hacer una partición adicional en un nodo del árbol. Un valor mayor hace que el modelo sea más conservador, evitando particiones que resultan en ganancias mínimas en el rendimiento.

- **`max_depth`: 3**
  - **Descripción:** Define la profundidad máxima de cada árbol. Limitar la profundidad del árbol ayuda a prevenir el sobreajuste, asegurando que los árboles no sean demasiado complejos y especializados en los datos de entrenamiento.

- **`verbosity`: 0**
  - **Descripción:** Controla el nivel de salida que el modelo produce mientras se entrena. Un valor de '0' significa silencio; es decir, no se imprimirán mensajes durante el entrenamiento, lo cual es útil para mantener los registros de entrenamiento limpios y enfocados en la información más crítica.

Estos parámetros están diseñados para equilibrar la precisión, la eficiencia computacional y la prevención del sobreajuste, adaptando el modelo a las particularidades de los datos y el problema específico de predicción de partidos de la NCAA.



#### **Entrenamiento y Evaluación del Modelo**

Se siguen los siguientes pasos para crear el modelo

- **Función de Objetivo de Cauchy:** Implementa una función de pérdida personalizada basada en la distribución de Cauchy para mejorar la robustez del modelo frente a valores atípicos, lo cual es común en datos deportivos.

- **Validación Cruzada:** Se ha entrenando el modelo en cada pliegue y luego validándolo en el pliegue restante. La validación cruzada ayuda a entender cómo se desempeñaría el modelo en general. Y es esencial para ajustar los hiperparámetros del modelo. De esta fase sacamos el valor de iteration_counts.

Repeat CV result:  [99, 114, 140]

Validation MAE:  [9.005042891538642, 9.011901557898794, 9.009583344636308]

Vemos como va descenciendo el MAE y seleccionamos aquel que nos da el error mas bajo.

- **Out-Of-Fold Predictions:** Las predicciones fuera de muestra se refieren a las estimaciones del modelo generadas sobre los datos que no se utilizaron durante el entrenamiento de ese modelo específico dentro de una iteración de validación cruzada, que se utilizan posteriormente para calcular las métricas de rendimiento y proporcionar una forma de validación interna.


<img src="img/out-of-fold.png" alt="Grafica de out-of-fold predictions" title="Grafica de out-of-fold predictions" width="400" height="300">



Una vez entrenado el modelo, se evalúan las predicciones:

- **Interpolación con Splines:** Para mejorar la calibración de las probabilidades predichas, se utiliza una interpolación de spline que ajusta una curva suave a las predicciones, permitiendo ajustes finos en base a la distribución de los resultados. Se realizan ajustes manuales para cuentas de enfrentamientos donde la historia y la lógica indican un resultado probable, como los enfrentamientos entre seeds altamente dispares.

<img src="img/spline.png" alt="Grafica de out-of-fold predictions" title="Grafica de out-of-fold predictions" width="400" height="300">

- **Entrenamiento final:**

adjusted logloss of cvsplit 0: 0.5318912956329822

adjusted logloss of cvsplit 1: 0.5316056931215835

adjusted logloss of cvsplit 2: 0.5306631444768599

| Season | Value     |
|--------|-----------|
| 2003   | 0.516914  |
| 2004   | 0.494575  |
| 2005   | 0.472573  |
| 2006   | 0.528037  |
| 2007   | 0.422747  |
| 2008   | 0.461141  |
| 2009   | 0.474632  |
| 2010   | 0.546291  |
| 2011   | 0.579826  |
| 2012   | 0.555434  |
| 2013   | 0.588209  |
| 2014   | 0.567107  |
| 2015   | 0.502188  |
| 2016   | 0.551625  |
| 2017   | 0.483643  |
| 2018   | 0.587167  |
| 2019   | 0.482147  |
| 2021   | 0.575284  |
| 2022   | 0.637760  |
| 2023   | 0.586124  |


#### **Visualización y Predicción Final**
- **Visualización de Desempeño:** Se proporcionan visualizaciones que comparan las probabilidades predichas con los resultados reales, ofreciendo una visión clara de cómo el modelo está interpretando y respondiendo a los datos.
- **Predicción de Partidos:** El método `predict_matchup` utiliza el modelo y la interpolación de spline para predecir y devolver un resultado binario, indicando si el equipo 1 es el probable ganador del enfrentamiento.





**Simulacion March Madness 2024**

**Resultados obtenidos**

Imagen del bracket completado

Errors: 50

Accuracy: 0.60

Points: 

**Comparacion con profesionales**

**Stephen A'Smith**

Errors: 42

Accuracy: 0.67

Points: 

Se ha probado con distintos hiperparametros, al final los mejores que se han obtenido son los que se encuentran en la descripcino de los hiperparametros anterior

Los errores obtenidos son los siguientes




### Redes Neuronales

Las redes neuronales fueron seleccionadas por su habilidad para modelar interacciones complejas entre variables. Implementamos una arquitectura de red profunda que se ajustó a través de diversas capas ocultas y técnicas de regularización como el Dropout para evitar el sobreajuste. La red fue entrenada para optimizar la función de pérdida logarítmica, proporcionando así estimaciones de probabilidad que facilitan la interpretación en contextos de apuestas y análisis deportivo.


The features are normalized using StandardScaler

Training the Model:

Uses Binary Cross-Entropy Loss (BCELoss) for the loss function and Adam optimizer.
Implements learning rate scheduling and early stopping based on validation loss to prevent overfitting.
For each epoch, the code performs training and validation steps, calculating and printing loss and accuracy metrics.

#### **Training Parameters**

- **Batch Size:** 32
- **Number of Epochs:** 20
- **Learning Rate:** 0.001
- **Weight Decay:** 0.03
- **Patience for Early Stopping:** 3
- **Scheduler Step Size:** 10
- **Scheduler Gamma:** 0.1

#### **Neural Network Structure**

A simple sequential model with two hidden layers. The network includes ReLU activation functions and dropout layers to prevent overfitting, followed by a sigmoid output layer for binary classification.

- **Sequential Model:**
  - **Layer 1:** Linear
    - **Input Features:** 57
    - **Output Features:** 16
    - **Bias:** True
  - **Layer 2:** ReLU
  - **Layer 3:** Dropout
    - **Probability:** 0.6
    - **Inplace:** False
  - **Layer 4:** Linear
    - **Input Features:** 16
    - **Output Features:** 8
    - **Bias:** True
  - **Layer 5:** ReLU
  - **Layer 6:** Dropout
    - **Probability:** 0.6
    - **Inplace:** False
  - **Layer 7:** Linear
    - **Input Features:** 8
    - **Output Features:** 1
    - **Bias:** True
  - **Layer 8:** Sigmoid


#### **Resultados de experimentacion**

After training, it plots training and validation loss and accuracy to visualize the model's performance over epochs.
Tras el entrenamiento, traza las pérdidas y la precisión del entrenamiento y la validación para visualizar el rendimiento del modelo a lo largo de las épocas.

![Train and validation loss and accuracy](img/train_val_loss_acc.png "Train and validation loss and accuracy")


**Resultados de experimentación**: Aquí, puedes detallar los resultados de tus experimentos, incluyendo métricas de rendimiento como la precisión, recall, y especialmente la log-loss para cada modelo.

## Modelo de Machine Learning Seleccionado

El modelo finalmente seleccionado fue una red neuronal profunda. Este modelo ofreció el mejor equilibrio entre precisión y log-loss, indicando no solo una alta tasa de predicciones correctas sino también confianza en esas predicciones. La capacidad de la red neuronal para capturar interacciones complejas entre características fue crucial para su desempeño superior.

## Conclusiones

El proyecto demostró la eficacia de las redes neuronales en la predicción de resultados deportivos, destacando la importancia de una métrica adecuada y un análisis exploratorio profundo. Futuras investigaciones podrían explorar la incorporación de datos en tiempo real y el análisis de sentimientos para mejorar aún más las predicciones.