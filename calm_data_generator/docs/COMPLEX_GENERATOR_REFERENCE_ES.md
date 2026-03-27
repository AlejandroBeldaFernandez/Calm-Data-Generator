# ComplexGenerator - Referencia

**Ubicacion:** `calm_data_generator.generators.complex.ComplexGenerator`

`ComplexGenerator` es una capa abstracta intermedia entre `BaseGenerator` y los generadores de dominio especificos (Clinico, Finanzas, IoT, Seguros). Expone tres motores matematicos reutilizables basados en la teoria de Copulas Gaussianas y modelado de efectos estocasticos, para que los nuevos generadores de dominio puedan heredarlos sin duplicar codigo.

---

## Jerarquia de Herencia

```
BaseGenerator (ABC)
└── ComplexGenerator          ← motores matematicos (este modulo)
    └── ClinicalDataGenerator ← dominio clinico
    └── FinanceGenerator      ← (ejemplo: definido por el usuario)
    └── IoTGenerator          ← (ejemplo: definido por el usuario)
```

---

## Inicio Rapido: Crear un Generador de Dominio

```python
from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator
import scipy.stats as stats
import numpy as np
import pandas as pd

class FinanceGenerator(ComplexGenerator):
    def generate(self, n_samples, n_assets, correlation_matrix=None, stress_events=None):
        if correlation_matrix is None:
            correlation_matrix = np.identity(n_assets)

        marginals = [stats.norm(loc=0.001, scale=0.02) for _ in range(n_assets)]
        X = self._generate_correlated_module(n_samples, marginals, correlation_matrix)
        df = pd.DataFrame(X, columns=[f"ASSET_{i}" for i in range(n_assets)])

        if stress_events:
            for event in stress_events:
                self.apply_stochastic_effects(df, df.index[event["rows"]], event)

        return df
```

---

## Motor 1: `_generate_correlated_module`

Genera datos correlacionados usando una Copula Gaussiana incondicional. Gestiona automaticamente matrices de correlacion no semidefinidas positivas (no-PSD) mediante reparacion de autovalores.

### Firma

```python
def _generate_correlated_module(
    self,
    n_samples: int,
    marginals_list: list,      # lista de distribuciones scipy congeladas
    sigma_module: np.ndarray,  # matriz de correlacion (n_vars x n_vars)
) -> np.ndarray:               # forma: (n_samples, n_vars)
```

### Parametros

| Parametro | Tipo | Descripcion |
|-----------|------|-------------|
| `n_samples` | int | Numero de muestras a generar |
| `marginals_list` | list | Distribuciones scipy congeladas, una por variable |
| `sigma_module` | np.ndarray | Matriz de correlacion. Las matrices no-PSD se reparan via `scipy.linalg.eigh` |

### Como funciona

1. Muestrea una normal multivariante `Z ~ N(0, sigma)` en el espacio latente
2. Aplica `U = Phi(Z)` (CDF normal estandar) para obtener marginales uniformes en [0,1]
3. Aplica `X_i = F_i^{-1}(U_i)` (PPF de cada marginal) para obtener la distribucion objetivo

### Ejemplo

```python
from scipy import stats

marginals = [
    stats.norm(loc=5, scale=1),       # variable 1: normal
    stats.expon(scale=2),             # variable 2: exponencial
    stats.lognorm(s=0.5, scale=10),   # variable 3: lognormal
]
correlacion = np.array([[1.0, 0.6, 0.3],
                         [0.6, 1.0, 0.4],
                         [0.3, 0.4, 1.0]])

X = gen._generate_correlated_module(500, marginals, correlacion)
# X.shape == (500, 3)
```

---

## Motor 2: `_generate_conditional_data`

Genera variables objetivo condicionadas a datos ya observados usando una Copula Gaussiana condicional. Soporta marginales de condicionamiento continuas y discretas (via Residuos Cuantilicos Aleatorizados).

### Firma

```python
def _generate_conditional_data(
    self,
    n_samples: int,
    conditioning_data: np.ndarray,      # (n_samples, n_cond)
    conditioning_marginals: list,        # marginales para variables de condicionamiento
    target_marginals: list,              # marginales para variables objetivo
    full_covariance: np.ndarray,         # (n_cond + n_target, n_cond + n_target)
) -> np.ndarray:                         # forma: (n_samples, n_target)
```

### Parametros

| Parametro | Tipo | Descripcion |
|-----------|------|-------------|
| `n_samples` | int | Numero de muestras |
| `conditioning_data` | np.ndarray | Datos observados para condicionar, forma `(n_samples, n_cond)` |
| `conditioning_marginals` | list | Marginales de las variables de condicionamiento. Las distribuciones discretas (binom, poisson, etc.) se manejan via RQR |
| `target_marginals` | list | Marginales de las variables a generar |
| `full_covariance` | np.ndarray | Covarianza conjunta sobre todas las variables (condicionamiento + objetivo) |

### Como funciona

1. Transforma `conditioning_data` al espacio latente usando CDFs marginales (con RQR para discretas)
2. Particiona la matriz de covarianza en `S_cc`, `S_ct`, `S_tc`, `S_tt`
3. Calcula media y covarianza condicionales: `mu_t|c = S_tc S_cc^{-1} Z_c`, `Sigma_t|c = S_tt - S_tc S_cc^{-1} S_ct`
4. Muestrea `Z_target ~ N(mu_t|c, Sigma_t|c)` por muestra
5. Aplica PPFs de las marginales objetivo para obtener los valores finales

### Ejemplo: Generar expresion genica condicionada a datos demograficos

```python
demographic_data = df[["edad", "imc"]].values  # (100, 2)
demographic_marginals = [stats.norm(50, 10), stats.norm(25, 4)]
gene_marginals = [stats.lognorm(s=0.5) for _ in range(20)]
joint_cov = np.eye(22)  # 2 demo + 20 genes

X_genes = gen._generate_conditional_data(
    n_samples=100,
    conditioning_data=demographic_data,
    conditioning_marginals=demographic_marginals,
    target_marginals=gene_marginals,
    full_covariance=joint_cov,
)
# X_genes.shape == (100, 20)
```

---

## Motor 3: `apply_stochastic_effects`

Aplica un efecto estocastico a un subconjunto de entidades en-lugar. Soporta 7 tipos de efectos para modelar senales de enfermedad, choques de mercado, anomalias de sensores, etc.

### Firma

```python
def apply_stochastic_effects(
    self,
    df: pd.DataFrame,     # modificado en-lugar
    entity_ids,           # etiquetas de indice de entidades a afectar
    effect_config: dict,  # ver abajo
) -> None:
```

### Configuracion del Efecto

```python
effect_config = {
    "index": [0, 1, 5],           # indices de columnas a afectar (int, lista o slice)
    "effect_type": "fold_change",  # uno de los 7 tipos indicados abajo
    "effect_value": 2.0,           # escalar, lista [bajo, alto], o dict (solo sigmoid)
}
```

Cuando `effect_value` es una lista `[bajo, alto]`, cada entidad recibe una muestra independiente de `Uniform(bajo, alto)`. Cuando es un escalar, las entidades reciben muestras independientes de `Normal(valor, |valor| * 0.1)`.

### Tipos de Efectos Soportados

| Tipo de Efecto | Formula | Caso de Uso |
|----------------|---------|-------------|
| `additive_shift` | `x += offset` | Senal de fondo, sesgo de sensor |
| `fold_change` | `x *= factor` | Sobreexpresion genica, multiplicador de precio |
| `power_transform` | `x **= exponente` | Distorsion no lineal |
| `variance_scale` | Reescala alrededor de la media | Heterocedasticidad, regimenes de volatilidad |
| `log_transform` | `x = log(x + eps)` | Logaritmizacion |
| `polynomial_transform` | `x = P(x)` | Transformacion polinomial (coeficientes en value) |
| `sigmoid_transform` | `x = 1/(1+e^{-k(x-x0)})` | Saturacion / recorte |

`simple_additive_shift` se acepta como alias de `additive_shift`.

### Ejemplo: Choque de estres en rentabilidades de activos

```python
stress_event = {
    "index": slice(0, 5),           # afectar los primeros 5 activos
    "effect_type": "additive_shift",
    "effect_value": [-0.1, -0.02],  # perdida aleatoria en [-10%, -2%]
}
filas_afectadas = df.index[:20]     # primeros 20 pasos temporales
gen.apply_stochastic_effects(df, filas_afectadas, stress_event)
```

### Notas

- Modifica `df` **en-lugar** (sin valor de retorno)
- Pasar `entity_ids` vacio es una operacion segura sin efecto
- Un `effect_type` desconocido lanza `ValueError`

---

## Gestion de Errores

| Situacion | Comportamiento |
|-----------|----------------|
| `sigma_module` no-PSD | Reparado via recorte de autovalores (sin excepcion) |
| `S_cc` singular en condicional | Regularizado con `+1e-6 * I` |
| Covarianza condicional no-PSD | Reparado via recorte de autovalores |
| Incompatibilidad de formas en `_generate_conditional_data` | Lanza `ValueError` |
| `effect_type` desconocido | Lanza `ValueError` |
| `entity_ids` vacio | Sin operacion, retorna inmediatamente |
