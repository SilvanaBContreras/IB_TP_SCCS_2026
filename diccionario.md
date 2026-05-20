# Diccionario de datos — Banco Ánfora v6

> Banco Ánfora · sistema de detección de fraude del APFLA.
> **EJERCICIO DIDÁCTICO. DATOS SIMULADOS.** Banco Ánfora y todas las
> entidades nombradas son construcciones ficticias.

## Archivos

### Operaciones

| Archivo | Filas | Descripción |
|---|---|---|
| `operaciones.parquet` | ~1.200.000 | Todas las operaciones de los 24 meses (T0 + T1 + T1' + T2), con etiquetas T0 al 100% y T1+T1'+T2 al 30%. |
| `operaciones_T0_train.parquet` | ~440.000 | 70% de T0 — datos de entrenamiento de los 5 modelos. |
| `operaciones_T0_valid.parquet` | ~190.000 | 30% de T0 — validación, base para F1 T0 y selección de threshold. |
| `operaciones_T1.parquet` | ~600.000 | Periodo post-smishing (meses 13-24), todas las operaciones. Incluye T2 (segundo drift en meses 22-24). |
| `operaciones_T1_etiquetadas.parquet` | ~180.000 | Subset 30% etiquetado de T1+T1'+T2 (auditoría). |

### Esquema de operaciones

| Columna | Tipo | Descripción |
|---|---|---|
| `op_id` | int | Identificador único de operación. |
| `mes` | int (1-24) | Mes desde el inicio del seguimiento. Meses 1-12 = T0 (2024). Meses 13-18 = T1 (1S 2025, drift smishing). Meses 19-21 = T1' (2S 2025, smishing estable). Meses 22-24 = T2 (1T 2026, segundo modus operandi sobre `monto_log`). |
| `canal` | str | `presencial`, `online`, `atm`, `transferencia`. |
| `hora_bin` | str | `madrugada`, `manana`, `tarde`, `noche`. |
| `dispositivo` | str | `conocido`, `registrado_reciente`, `desconocido`. |
| `monto_log` | float | log(1 + monto en pesos). |
| `terminal_risk_score` | float | Score histórico de riesgo de la terminal $[0, 1]$. Calibrado por el sistema interno con `delay` de 7 días respecto a la observación. |
| `secuencia_24h` | int | Cantidad de operaciones del mismo cliente en las últimas 24 hs. |
| `etiquetado` | bool | `True` si la operación tiene etiqueta de fraude conocida. |
| `es_fraude` | bool | Etiqueta de fraude (válida si `etiquetado=True`). |

### Modelos pre-entrenados

| Archivo | Tipo | Features | Threshold | F1 validación T0 |
|---|---|---|---|---|
| `modelos/M1_logistica.pkl` | LogisticRegression | monto_log + canal + dispositivo (3) | 0,14 | 0,243 |
| `modelos/M2_random_forest.pkl` | RandomForestClassifier | + hora_bin + secuencia_24h (5) | 0,11 | 0,243 |
| `modelos/M3_gbm_completo.pkl` | GradientBoostingClassifier | + terminal_risk_score (6) | 0,46 | **0,842** |
| `modelos/M4_gbm_moderado.pkl` | GradientBoostingClassifier | mismas que M2 (5) | 0,11 | 0,236 |
| `modelos/M5_naive_bayes.pkl` | GaussianNB | monto_log + secuencia_24h (2) | 0,11 | 0,214 |

(F1 T0 reportado sobre la validación T0 entera. El F1 mensual promedio
de M3 sobre los 12 meses de T0 es ≈ 0,874; la pequeña diferencia con
0,842 viene de promediar particiones mensuales versus computar F1 sobre
toda la validación de una vez.)

Cada `.pkl` contiene un `dict` con keys `pipeline` (sklearn Pipeline ya con preprocesamiento), `features` (lista de columnas) y `threshold` (umbral óptimo en validación T0).

Uso típico:

```python
import pickle
with open("modelos/M3_gbm_completo.pkl", "rb") as f:
    obj = pickle.load(f)
proba = obj["pipeline"].predict_proba(df[obj["features"]])[:, 1]
pred = (proba >= obj["threshold"]).astype(int)
```

`metricas_T0_validacion.json` contiene precision, recall, F1, AUC-ROC y AUC-PR de cada modelo en validación T0, además de la descripción y los features de cada uno.

### Métricas temporales

`metricas_temporales.csv` (120 filas: 5 modelos × 24 meses) reporta F1,
precision, recall y AUC-PR mensuales con bootstrap percentile-CI 95% por
celda.

> **Nota metodológica.** Las celdas mensuales de este CSV usan
> *bootstrap percentile-CI* (200 resamples) sobre F1 puntual, suficiente
> para la figura panel y para la trayectoria temporal. Las dos lecturas
> (bootstrap mensual y posteriores agregadas con conjugación Beta-Binomial)
> son consistentes en el bulk pero no idénticas: el bootstrap es una
> aproximación frecuentista, mientras que la posterior conjugada admite
> una formulación exacta. El bootstrap mensual está en el CSV como
> insumo visual; las posteriores propiamente dichas las debe construir
> el área externa que evalúe el sistema sobre los datos etiquetados que
> elija como ventana informativa.

Esquema de `metricas_temporales.csv`:

| Columna | Tipo | Descripción |
|---|---|---|
| `modelo` | str | M1_logistica, M2_random_forest, M3_gbm_completo, M4_gbm_moderado, M5_naive_bayes. |
| `mes` | int | 1-24. |
| `fase` | str | T0 (1-12), T1 (13-21, incluye T1' meses 19-21) o T2 (22-24, segundo modus operandi sobre `monto_log`). |
| `n` | int | Operaciones etiquetadas en el mes. |
| `n_fraude` | int | Operaciones positivas en el mes. |
| `f1` | float | F1 puntual con threshold del modelo. |
| `f1_med` | float | Mediana bootstrap de F1. |
| `f1_lo`, `f1_hi` | float | Percentiles 2,5 y 97,5 del bootstrap (CI 95%). |
| `precision`, `recall`, `auc_pr` | float | Métricas adicionales. |

## Notas metodológicas

- **Muestreo de T1 etiquetado**: el 30% de T1 etiquetado simula auditoría humana de un subset aleatorio. La distribución no es uniforme entre meses pero la cantidad por mes es suficiente para HPDI 95%.
- **Threshold por modelo**: cada modelo tiene su umbral óptimo guardado, calibrado para maximizar F1 sobre validación T0. El threshold no se reoptimiza en T1.
- **Reproducibilidad**: semilla maestra `20260424`. Pipeline completo end-to-end: `gen_operaciones.py` → `train_modelos.py` → `evaluar_temporal.py` → `analisis_bayesiano.py`. Tiempo total ~3 minutos en laptop estándar.
- **`terminal_risk_score`**: feature mantenida por el sistema interno del banco con un `delay` de 7 días respecto a la observación. Su comportamiento bajo distintos regímenes operativos es uno de los temas a evaluar por el análisis externo.

## Banner

> EJERCICIO DIDÁCTICO — DATOS SIMULADOS. Banco Ánfora, sus personajes,
> transacciones y escenarios son completamente ficticios. Uso exclusivo
> académico.
