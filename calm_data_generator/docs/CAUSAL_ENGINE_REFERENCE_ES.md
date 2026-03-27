# CausalEngine - Referencia

**Ubicacion:** `calm_data_generator.generators.dynamics.CausalEngine`

`CausalEngine` implementa propagacion de cascadas causales basada en grafos dirigidos aciclicos (DAG). Permite definir grafos de dependencias entre variables y propagar una perturbacion desde una variable disparadora hacia todos sus descendientes, usando funciones de transferencia definidas por el usuario (lineal, exponencial, potencia, polinomial o cualquier callable).

---

## Inicio Rapido

```python
from calm_data_generator.generators.dynamics.CausalEngine import CausalEngine
import numpy as np
import pandas as pd

# Cadena IoT: temperatura -> presion -> tasa de fallo del sensor
dag = {
    "temperatura": [],
    "presion": [
        {"parent": "temperatura", "func": "linear", "params": {"slope": 1.2}}
    ],
    "tasa_fallo": [
        {"parent": "presion", "func": "exponential", "params": {"scale": 0.001, "rate": 0.3}}
    ],
}

engine = CausalEngine(dag)

df = pd.DataFrame({
    "temperatura": np.random.uniform(20, 40, 500),
    "presion":     np.random.uniform(1, 5, 500),
    "tasa_fallo":  np.zeros(500),
})

# Aplicar un choque de +5 grados a todas las filas
df_shock = engine.apply_cascade(df.copy(), trigger_col="temperatura", delta=np.full(500, 5.0))
```

---

## Formato del DAG

```python
dag_config = {
    "nombre_nodo": [                       # lista de aristas padre (vacia = nodo raiz)
        {
            "parent": "columna_padre",     # requerido: nombre del nodo padre
            "func":   "linear",            # requerido: funcion de transferencia (ver tabla)
            "params": {"slope": 1.2},      # opcional: parametros de la funcion
        }
    ],
}
```

Un nodo con lista vacia `[]` es una **raiz** (sin padres). Los nodos pueden tener multiples padres — sus contribuciones de delta se suman.

---

## Funciones de Transferencia

La propagacion usa un enfoque **diferencial**:
```
delta_hijo = f(v_padre + delta_padre) - f(v_padre)
```
Esto preserva los valores absolutos actuales y solo propaga el cambio incremental.

| `func` | Formula | Parametros |
|--------|---------|------------|
| `"linear"` | `slope * delta_padre` | `slope` (default 1.0), `intercept` se ignora en deltas |
| `"exponential"` | `scale * (exp(rate*(v+d)) - exp(rate*v))` | `scale` (default 1.0), `rate` (default 1.0) |
| `"power"` | `scale * ((v+d)^exp - v^exp)` | `scale` (default 1.0), `exponent` (default 2.0) |
| `"polynomial"` | `P(v+d) - P(v)` | `coeffs`: lista de coeficientes para `np.poly1d` |
| callable | `func(v+d) - func(v)` | — |

---

## Referencia de API

### `CausalEngine(dag_config)`

```python
engine = CausalEngine(dag_config: dict)
```

Valida el DAG en la construccion (detecta ciclos via algoritmo de Kahn). Lanza `ValueError` si:
- Se detecta un ciclo
- Una arista referencia un nodo no declarado como clave en `dag_config`

### `apply_cascade(df, trigger_col, delta, rows=None)`

```python
df_result = engine.apply_cascade(
    df: pd.DataFrame,
    trigger_col: str,          # debe ser un nodo del DAG
    delta: np.ndarray,         # vector de perturbacion (longitud = len(rows) o len(df))
    rows: pd.Index = None,     # None = todas las filas
) -> pd.DataFrame              # df modificado en-lugar, tambien retornado
```

Aplica `delta` a `trigger_col` y propaga a todos sus descendientes en orden topologico.

### `get_topological_order()`

Retorna todos los nodos como lista ordenada de raices a hojas. Util para inspeccion y tests.

### `validate()`

Re-valida el DAG. Lanza `ValueError` en ciclos o referencias a nodos desconocidos.

---

## Uso con DriftInjector

`DriftInjector.inject_causal_cascade` envuelve `CausalEngine` e integra el sistema completo de seleccion de filas y reportes:

```python
from calm_data_generator.generators.drift.DriftInjector import DriftInjector

injector = DriftInjector(original_df=df)

dag = {
    "temperatura": [],
    "presion":    [{"parent": "temperatura", "func": "linear",      "params": {"slope": 0.5}}],
    "tasa_fallo": [{"parent": "presion",     "func": "exponential", "params": {"scale": 0.01, "rate": 0.3}}],
}

# Aplicar cascada solo a filas donde temperatura > 35
df_result = injector.inject_causal_cascade(
    df,
    dag_config=dag,
    trigger_col="temperatura",
    trigger_delta=10.0,
    conditions=[{"column": "temperatura", "operator": ">", "value": 35}],
)
```

---

## Uso con ScenarioInjector

`evolve_type: "driven_by"` hace que una feature evolucione proporcionalmente al valor actual de otra columna en cada fila:

```python
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector

si = ScenarioInjector()
df_evolved = si.evolve_features(df, evolution_config={
    "presion": {
        "type":        "driven_by",
        "driver_col":  "temperatura",
        "func":        "linear",
        "func_params": {"slope": 0.8},
    }
})
# Cada fila: delta_presion = 0.8 * valor_temperatura
```

---

## Ejemplos por Dominio

### Finanzas: propagacion de choque de precios

```python
dag = {
    "indice_mercado": [],
    "etf_sector": [
        {"parent": "indice_mercado", "func": "linear", "params": {"slope": 1.1}}
    ],
    "accion_individual": [
        {"parent": "indice_mercado", "func": "linear",      "params": {"slope": 0.6}},
        {"parent": "etf_sector",     "func": "exponential", "params": {"scale": 0.3, "rate": 0.05}},
    ],
}
engine = CausalEngine(dag)
df_shock = engine.apply_cascade(df, "indice_mercado", delta=np.full(len(df), -0.05))
```

### Clinico: cascada de efecto de tratamiento

```python
dag = {
    "dosis_farmaco": [],
    "biomarcador_A": [
        {"parent": "dosis_farmaco", "func": "linear", "params": {"slope": -0.3}}
    ],
    "biomarcador_B": [
        {"parent": "biomarcador_A", "func": "power", "params": {"scale": 1.0, "exponent": 1.5}}
    ],
}
```

---

## Gestion de Errores

| Situacion | Comportamiento |
|-----------|----------------|
| Ciclo en el DAG | `ValueError` en el momento de la construccion |
| Arista a nodo desconocido | `ValueError` en el momento de la construccion |
| `trigger_col` no esta en el DAG | `ValueError` en `apply_cascade` |
| Nodo en el DAG pero no en df | Ignorado silenciosamente (delta calculado pero no aplicado) |
| Nodo en df pero no en el DAG | No afectado por la cascada |
