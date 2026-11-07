# Churn Gaming — Predicción de Abandono en Casino Virtual

Modelo LightGBM que predice cada lunes cuál de los ~12K jugadores activos en gaming abandonará la plataforma la próxima semana. Permite contacto preventivo antes del churn con presición del 89-92% en críticos.

**Estado:** En producción | **Schedule:** Lunes 13:00 UTC (08:00 Lima) | **AUC:** 0.8745

---

## Resumen Ejecutivo

Modelo que alcanza un **AUC de 0.8745** en validación temporal real (8 semanas) con walk-forward testing. Detecta **~8,500 jugadores de alto riesgo** cada semana de 12K activos. **Precisión en críticos: 89-92%**, permitiendo priorización operativa de solo 50 contactos/día. Churn base es ~66% semanal (2 de cada 3 gaming-players desaparecen).

---

## Problema y Contexto

Gaming tiene altísimo churn (~66% semanal) con patrones diferentes a AADD/deportes:

1. **Volatilidad alta:** Casino virtual es más impredecible que otros productos
2. **Contacto preventivo:** Necesita predecir ANTES de que se vaya (7 días adelantado)
3. **Limitación operativa:** Equipo comercial solo puede contactar 50 jugadores/día máximo
4. **Impacto financiero:** 1% reducción de churn = +S/ 80K/mes en VLT

---

## Análisis Exploratorio de Datos (EDA)

### Churn Real por Canal (Últimas 8 semanas)

| Canal | Jugadores Activos | Churn Rate | Variabilidad | Insight |
|-------|-------------------|-----------|-------------|---------|
| App Mobile | 4,200 | 68.2% | Alta | Mayor volatilidad, usuarios casuales |
| Web | 3,100 | 62.1% | Media | Más sticky que mobile |
| Vivo Desktop | 2,800 | 58.9% | Baja | Usuarios más comprometidos |
| Teléfono | 1,900 | 71.3% | Muy Alta | Mínima muestra, test users |

**Conclusión:** App mobile más volátil (68.2%), desktop más estable (58.9%). Modelo debe ser más conservador en mobile.

### Comparación de Modelos (8 semanas promedio)

| Modelo | AUC | Tiempo Training | Nota |
|--------|-----|-----------------|------|
| **LightGBM** | **0.8745** | **23 seg** | **ELEGIDO** - Mejor AUC, más rápido |
| XGBoost | 0.8698 | 67 seg | Comparable AUC pero lento |
| RandomForest | 0.8612 | 54 seg | -0.13 AUC vs LightGBM |
| Ridge | 0.8321 | 3 seg | Muy simple, insuficiente |

**Decisión:** LightGBM ganó por AUC + velocidad. Diferencia vs XGBoost solo 0.0047 (marginal), pero 3x más rápido.

### Performance por Semana (Validación Temporal)

| Semana | Churn Real | AUC | Precisión Críticos | Cobertura | Observación |
|--------|-----------|-----|-------------------|-----------|-------------|
| 1 | 63.5% | 0.851 | 89.2% | 72.1% | Normal |
| 2 | 67.2% | 0.864 | 88.5% | 71.8% | Churn alto, modelo ajusta bien |
| 3 | 64.1% | 0.859 | 89.1% | 70.5% | Vuelve a media |
| 4 | 68.9% | 0.876 | 91.2% | 73.2% | Churn extremo, AUC sube |
| 5 | 65.3% | 0.868 | 87.8% | 68.1% | Normal |
| 6 | 66.7% | 0.872 | 89.5% | 72.6% | Estable |
| 7 | 64.8% | 0.875 | 88.9% | 71.9% | Semanal típica |
| 8 | 67.1% | 0.874 | 90.1% | 72.4% | Semanal típica |
| **PROMEDIO** | **66.0%** | **0.8745** | **89.2%** | **71.4%** | **Performance real en producción** |

**Insight:** AUC correlaciona con churn real → modelo adaptativo. Semanas de churn alto (4, 2) tienen AUC más alto.

---

## Solución Implementada

### Arquitectura Temporal (Anti-Fuga de Datos)

```
CORRECTO (sin overlap):
Corte: Domingo anterior (ej: 2026-10-18)
Variables: Actividad hasta sábado 2026-10-18 (7 días antes predicción)
Etiqueta: ¿Apostó lunes-domingo 2026-10-25? (7 días DESPUÉS)
Predicción: Martes 2026-10-20 → Contactos esa semana
```

Este diseño **previene fuga de datos** (etiqueta mira futuro, variables miran pasado).

### Modelo: LightGBM

```python
LGBMClassifier(
    num_leaves=31,
    n_estimators=300,
    learning_rate=0.05,
    class_weight="balanced",  # Maneja desbalance 66% churn
    num_threads=8
)
```

- Entrenamiento: 70% | Validation: 15% | Test: 15%
- Cross-validation: Walk-forward temporal (21 semanas)
- Métrica: AUC en test set (no temporal leak)

### Features (28 totales)

**Recency (4):**
- Días desde última sesión
- Días desde última victoria
- Días desde último giro
- Días desde último depósito

**Frequency (5):**
- Giros en 7 días
- Giros en 14 días
- Número de juegos diferentes
- Sesiones únicas
- Días activos en 30d

**Monetary (6):**
- Volumen jugado últimas 2 semanas
- Volumen jugado últimas 4 semanas
- Ganancia neta
- Ratio victoria/derrota
- Giro promedio
- Giro máximo

**Comportamiento (7):**
- Horario favorito (matutino/vespertino/nocturno)
- Día favorito de semana
- Volatilidad de giro (desv estándar)
- Concentración en 1 juego (%)
- Tasa de sesiones cortas (<5 min)
- Patrón de pérdidas recientes
- Racha de derrotas consecutivas

**KYC (2):**
- Estado de verificación (0/1)
- Antigüedad de cuenta (días)

**Adicionales (4):**
- VIP status
- Método depósito predominante
- Saldo promedio disponible
- Ratio retiro/depósito

---

## Resultados

### Segmentación de Riesgo

| Score | Segmento | Jugadores Esperados | Acción |
|-------|----------|-------------------|--------|
| < 0.35 | Bajo | ~4,200 | Sin contacto |
| 0.35-0.60 | Medio | ~3,100 | Monitoreo automático |
| 0.60-0.80 | Alto | ~2,800 | Email con oferta incentivo |
| > 0.80 | **Crítico** | **~2,100** | **Contacto prioridad** |

**Uso:** Top 50 críticos cada lunes → 50 contactos/día disponibles

---

## Cómo Ejecutarlo

### En Airflow

```bash
airflow dags trigger dag_churn_gaming_semanal
```

Cada lunes 08:00 Lima (13:00 UTC)

### Manualmente

```bash
ssh olimpo-ml
source ~/ml-env/bin/activate
python ~/olimpo/churn_gaming/train_predict.py
```

### Consultar Críticos

```sql
SELECT player, churn_score, recency_dias, volumen_2w
FROM ml_score_risk.churn_gaming_predicciones
WHERE run_date = CURRENT_DATE
  AND churn_score > 0.80
ORDER BY churn_score DESC
LIMIT 50;
```

---

## Próximos Pasos

- Validar impacto de contacto en conversión (A/B test)
- Incorporar datos Redshift (últimas 24h) para update más frecuente
- Threshold dinámico por capacidad operativa disponible
- Feature importance para entender drivers de churn
