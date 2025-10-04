# =========================
# 0) Imports (mantener)
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# =========================
# 1) Cargar, índice e INFO
# =========================
loan_df = pd.read_csv("personal_loan.csv")
loan_df.set_index("ID", inplace=True)

print(loan_df.info())
original_len = len(loan_df)
print('Longitud del DataFrame (inicial):', original_len)

# ¿Cuántas columnas numéricas vs texto?
int_cols_ini   = loan_df.select_dtypes(include=["int64"]).columns.tolist()
float_cols_ini = loan_df.select_dtypes(include=["float64"]).columns.tolist()
obj_cols_ini   = loan_df.select_dtypes(include=["object"]).columns.tolist()
print("\nColumnas int64:", int_cols_ini, "Total:", len(int_cols_ini))
print("Columnas float64:", float_cols_ini, "Total:", len(float_cols_ini))
print("Columnas object:", obj_cols_ini, "Total:", len(obj_cols_ini))

# =========================
# 2) Descriptivas numéricas + limpieza
# =========================

# 2.1 Eliminar edades no plausibles (>100)
print("\nRegistros con Age > 100 (pre-eliminación):", (loan_df["Age"] > 100).sum())
loan_df = loan_df[loan_df["Age"] <= 100]

# 2.2 Unificar binarios a 0/1 (acepta 'yes'/'no' o '1'/'0' en texto)
yn_cols = ["Personal Loan", "Securities Account", "CD Account", "Online", "CreditCard"]
for col in yn_cols:
    loan_df[col] = loan_df[col].astype(str).str.strip().str.lower().map({
        "yes": 1, "no": 0, "1": 1, "0": 0
    })
# quitar filas con valores raros no mapeables
loan_df = loan_df.dropna(subset=yn_cols)
for col in yn_cols:
    loan_df[col] = loan_df[col].astype(int)

# 2.3 Otras validaciones simples
loan_df = loan_df[loan_df["Experience"] >= 0]   # experiencia negativa no plausible
loan_df = loan_df[loan_df["Family"] >= 0]       # tamaño de familia negativo no plausible

# 2.4 Education debe ser 1, 2 o 3 (entero)
loan_df["Education"] = loan_df["Education"].round().astype(int)
loan_df = loan_df[loan_df["Education"].isin([1, 2, 3])]

# Reporte tras limpieza (solo paso 2)
after_clean_len = len(loan_df)
elim_clean = original_len - after_clean_len
pct_clean = (elim_clean / original_len) * 100
print("\nPrimeros registros después de limpiar (paso 2):")
print(loan_df.head(10))
print('Longitud después de limpieza (paso 2):', after_clean_len)
print(f"Registros eliminados en paso 2: {elim_clean}  ({pct_clean:.2f}%)")

print("\nEstadísticas descriptivas (numéricas):")
print(loan_df.describe())

# =========================
# 3) Texto / categorías: descriptivas y frecuencias
# (ya unificamos binarios; imprime frecuencias)
# =========================
print("\nFrecuencias de variables binarias (0=No, 1=Sí):")
for col in yn_cols:
    print(f"\n{col}:\n", loan_df[col].value_counts(dropna=False))

# (Opcional) top 10 ZIP para no saturar
print("\nZIP Code (top 10):")
print(loan_df["ZIP Code"].value_counts().head(10))

# =========================
# 4) Duplicados y reiniciar índice
# =========================
duplicados = loan_df.duplicated().sum()
print("\nNúmero de registros duplicados:", duplicados)
if duplicados > 0:
    loan_df = loan_df.drop_duplicates()
    print("Duplicados eliminados.")

loan_df.reset_index(drop=True, inplace=True)

# Eliminados totales (paso 2 + duplicados)
final_len_so_far = len(loan_df)
elim_totales = original_len - final_len_so_far
pct_totales = (elim_totales / original_len) * 100
print(f"\nRegistros eliminados (incluyendo duplicados): {elim_totales}")
print(f"Porcentaje total eliminado: {pct_totales:.2f}%")

# =========================
# 5) Conversión de tipos según naturaleza + listas
# =========================
# ZIP Code como texto (nominal)
loan_df["ZIP Code"] = loan_df["ZIP Code"].astype(str)

# Education como categoría ordinal (1<2<3)
loan_df["Education"] = pd.Categorical(loan_df["Education"], categories=[1, 2, 3], ordered=True)

# Binarias como categoría (manteniendo 0/1)
for col in yn_cols:
    loan_df[col] = pd.Categorical(loan_df[col], categories=[0, 1])

# Listas de numéricas y categóricas
num_cols = loan_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = loan_df.select_dtypes(include=["object", "category"]).columns.tolist()
print("\nColumnas numéricas:", num_cols)
print("Columnas categóricas:", cat_cols)

# (Opcional) describe de object/category ya con tipos finales
obj_df = loan_df.select_dtypes(include=["object"])
cat_df = loan_df.select_dtypes(include=["category"])
print("\nDescribe de columnas object (texto):")
print(obj_df.describe())
print("\nDescribe de columnas category:")
print(cat_df.describe())


# =========================
# 6) Descriptivas numéricas con asimetría y curtosis
# =========================
# Resumen con skew (asimetría) y kurt (curtosis de Fisher; normal = 0)
num_summary = loan_df[num_cols].agg(
    ["count", "mean", "median", "std", "min", "max", "skew", "kurt"]
).T
print("\n[6] Resumen numérico con asimetría y curtosis:\n")
print(num_summary)

# Funciones de clasificación sencillas
def clasificar_asimetria(sk):
    if abs(sk) < 0.1:
        return "≈ simétrica"
    elif sk > 0:
        return "asimetría positiva (sesgo a la derecha)"
    else:
        return "asimetría negativa (sesgo a la izquierda)"

def clasificar_curtosis(ku):
    # Fisher: 0 ~ normal; >0 leptocúrtica; <0 platicúrtica
    if ku > 0.5:
        return "leptocúrtica (> 0)"
    elif ku < -0.5:
        return "platicúrtica (< 0)"
    else:
        return "≈ mesocúrtica (cercana a 0)"

print("\n[6] Clasificación para Age, Income y Mortgage:\n")
for col in ["Age", "Income", "Mortgage"]:
    sk = loan_df[col].skew()
    ku = loan_df[col].kurt()
    print(f"- {col}: {clasificar_asimetria(sk)} | {clasificar_curtosis(ku)} "
          f"(skew={sk:.3f}, kurt={ku:.3f})")


# =========================
# 7) Histogramas con KDE + Normal de referencia, y boxplots con media
# =========================
for col in num_cols:
    serie = loan_df[col].dropna()
    mu = serie.mean()
    sigma = serie.std()

    # --- Histograma + KDE + Normal(mu, sigma) ---
    plt.figure(figsize=(7, 4))
    sns.histplot(serie, bins="auto", stat="density", kde=True)
    # Curva normal de referencia (si hay variabilidad)
    if sigma > 0:
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        y = norm.pdf(x, loc=mu, scale=sigma)
        plt.plot(x, y, linewidth=2)  # línea de la normal
    plt.title(f"Histograma y KDE de {col}")
    plt.xlabel(col)
    plt.ylabel("Densidad")
    plt.tight_layout()
    plt.show()

    # --- Boxplot individual con línea de la media ---
    plt.figure(figsize=(6, 3.5))
    ax = sns.boxplot(y=serie)
    ax.axhline(mu, linestyle="--", linewidth=2, label=f"Media: {mu:.2f}")
    plt.title(f"Boxplot de {col}")
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Comentario automático: relación media vs mediana
    med = serie.median()
    if mu > med:
        comentario = "Media > mediana → sesgo a la derecha (asimetría positiva)."
    elif mu < med:
        comentario = "Media < mediana → sesgo a la izquierda (asimetría negativa)."
    else:
        comentario = "Media ≈ mediana → distribución aproximadamente simétrica."
    print(f"[7] {col}: media={mu:.2f}, mediana={med:.2f}. {comentario}")


# # =========================
# # 8) Descriptivas categóricas + barras (top 10 si hay mucha variedad)
# # =========================
print("\n[8] Describe de variables categóricas:\n")
print(loan_df[cat_cols].describe())

for col in cat_cols:
    vc = loan_df[col].value_counts(dropna=False)
    # Si hay muchas categorías, mostrar top 10
    if len(vc) > 10:
        vc = vc.head(10)
        titulo = f"{col} (Top 10)"
    else:
        titulo = col

    plt.figure(figsize=(7, 4))
    vc.plot(kind="bar")
    plt.title(f"Frecuencia de {titulo}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

# =========================
# 9) Scatter matrix y mapa de calor de correlaciones
# =========================
from pandas.plotting import scatter_matrix

# Scatter matrix (todas las numéricas)
plt.figure(figsize=(8, 8))
scatter_matrix(loan_df[num_cols], figsize=(10, 10), diagonal='hist')
plt.suptitle("Scatter Matrix - Variables numéricas", y=1.02)
plt.show()

# Correlaciones de Pearson
corr = loan_df[num_cols].corr(method='pearson')
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Mapa de calor - Correlación de Pearson (numéricas)")
plt.tight_layout()
plt.show()

# Elegir un par representativo y mostrar su correlación
# (ejemplos típicos: Income vs CCAvg, Age vs Experience)
par1 = ("Income", "CCAvg")
par2 = ("Age", "Experience")

c1 = loan_df[par1[0]].corr(loan_df[par1[1]])
c2 = loan_df[par2[0]].corr(loan_df[par2[1]])

print(f"[9] Correlación {par1[0]} vs {par1[1]}: {c1:.3f}")
print("¿El valor se corresponde con lo esperado al ver la nube de puntos y el mapa de calor?")
print("Sí, a mayor ingreso mayor gasto en tarjetas")

print(f"[9] Correlación {par2[0]} vs {par2[1]}: {c2:.3f}")
print("¿El valor se corresponde con lo esperado al ver la nube de puntos y el mapa de calor?")
print("Sí, a mayor mayor edad mayor experiencia")



# =========================
# 10) Análisis respecto a 'Personal Loan'
# =========================

# --- A) Numéricas: boxplots por clase de Personal Loan ---
for col in num_cols:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=loan_df, x="Personal Loan", y=col)
    plt.title(f"{col} vs Personal Loan")
    plt.xlabel("Personal Loan (0=No, 1=Sí)")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# Comentario rápido (diferencia de medias por clase)
print("\n[10-A] Diferencia de medias (Clase=1 menos Clase=0) en variables numéricas:")
means_by_target = loan_df.groupby("Personal Loan")[num_cols].mean()
diff_means = (means_by_target.loc[1] - means_by_target.loc[0]).sort_values(ascending=False)
print(diff_means)
print("Valores positivos: mayores en quienes aceptaron el préstamo (1). Negativos: mayores en (0).")

# --- B) Categóricas (sin ZIP Code): barras apiladas con distribución relativa de Personal Loan ---
cats_for_bars = [c for c in cat_cols if c not in ["ZIP Code", "Personal Loan"]]

for col in cats_for_bars:
    # distribución relativa del target dentro de cada categoría
    ct = pd.crosstab(loan_df[col], loan_df["Personal Loan"], normalize='index')  # filas suman 1
    ct = ct[[0, 1]] if 0 in ct.columns and 1 in ct.columns else ct  # asegurar orden si están ambas clases
    ax = ct.plot(kind="bar", stacked=True, figsize=(7, 4))
    plt.title(f"Distribución relativa de Personal Loan por {col}")
    plt.xlabel(col)
    plt.ylabel("Proporción")
    plt.legend(title="Personal Loan", labels=["0 (No)", "1 (Sí)"] if 0 in ct.columns and 1 in ct.columns else None)
    plt.tight_layout()
    plt.show()

    # tasa de aceptación por categoría (útil para comentar hallazgos)
    tasa = (
        loan_df.groupby(col)["Personal Loan"]
           .apply(lambda s: s.astype(int).mean())
           .sort_values(ascending=False)
    )
    print(f"[10-B] Tasa de aceptación (promedio de Personal Loan=1) por {col}:")
    print(tasa.head(10), "\n")

print("Comenta al menos un hallazgo por grupo (numéricas y categóricas) usando los gráficos y tablas impresas.")
print(". Age vs Personal Loan:")
print("   - La edad a la que las personas comienzan a sacar prestamos es despues de los 35")

print("1. Experience vs Personal Loan:")
print("   - La experiencia laboral es similar en clientes con y sin préstamo personal.")
print("   - No es un factor determinante en la decisión de préstamo.\n")

print("2. Income vs Personal Loan:")
print("   - Los clientes con préstamo tienen ingresos significativamente más altos.")
print("   - El ingreso es un factor importante en la aprobación del préstamo.\n")

print("3. Family vs Personal Loan:")
print("   - Los clientes sin préstamo tienden a tener familias más pequeñas (mediana ≈ 2).")
print("   - Los clientes con préstamo tienden a tener familias más grandes (mediana ≈ 3).")
print("   - El tamaño de la familia puede estar asociado con una mayor probabilidad de préstamo.\n")

print("4. CCAvg vs Personal Loan:")
print("   - Los clientes con préstamo muestran un gasto promedio con tarjeta mucho mayor.")
print("   - Existe una relación positiva entre gasto en tarjeta de crédito y préstamo personal.\n")

print("5. Mortgage vs Personal Loan:")
print("   - Existe mucha dispersión, en ambos grupos las hipotecas varían mucho y hay varios valores atípicos.")
print("   - Los clientes con préstamo tienden a hipotecas más altas, aunque no es un factor tan decisivo.\n")

print("Conclusión general:")
print("   - Income y CCAvg son los factores más diferenciadores.")
print("   - Family y Mortgage tienen un efecto moderado.")
print("   - Experience muestra poca relevancia en la decisión de otorgar un préstamo.")

print("6. Education vs Personal Loan:")
print("   - A mayor nivel educativo (2 y 3), aumenta la proporción de clientes con préstamo personal.")
print("   - Los clientes con educación básica (1) tienen una tasa muy baja de préstamos.\n")

print("7. Securities Account vs Personal Loan:")
print("   - Los clientes con cuenta de valores (Securities Account = 1) tienen una ligera mayor proporción de préstamos.")
print("   - Sin embargo, la diferencia no es tan grande respecto a quienes no tienen.\n")

print("8. CD Account vs Personal Loan:")
print("   - Los clientes con cuenta de certificados (CD Account = 1) tienen una proporción muy alta de préstamos.")
print("   - Esta variable parece ser un fuerte indicador en la aceptación del préstamo personal.\n")

print("9. Online vs Personal Loan:")
print("   - La diferencia entre clientes que usan servicios online y los que no es pequeña.")
print("   - El canal online no parece ser un factor muy determinante para tener un préstamo.\n")

print("10. CreditCard vs Personal Loan:")
print("   - Los clientes con tarjeta de crédito tienen una proporción ligeramente mayor de préstamos.")
print("   - Sin embargo, la diferencia es baja comparada con Education o CD Account.\n")

print("Conclusión general (categóricas):")
print("   - CD Account y Education muestran una fuerte relación con la tenencia de préstamo personal.")
print("   - Securities Account y CreditCard tienen un efecto moderado.")
print("   - Online muestra muy poca relevancia en la decisión de préstamo.")
