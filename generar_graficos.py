import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c', 'danger': '#d62728', 'warning': '#ff9800'}

def set_style(ax):
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Grafico 1: AUC Semanal
semanas = np.arange(1, 9)
auc_modelo = np.array([0.851, 0.864, 0.859, 0.876, 0.868, 0.872, 0.875, 0.874])

axes[0, 0].plot(semanas, auc_modelo, marker='o', linewidth=2.5, markersize=8,
               color=COLORS['success'], label='LightGBM', markeredgecolor='black', markeredgewidth=1.5)
axes[0, 0].axhline(y=0.8745, color=COLORS['danger'], linestyle='--', linewidth=2, label='Promedio (0.8745)')
axes[0, 0].set_xlabel('Semana', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('AUC', fontsize=11, fontweight='bold')
axes[0, 0].set_title('AUC Semanal: 8 semanas', fontsize=12, fontweight='bold', pad=15)
axes[0, 0].set_ylim([0.84, 0.88])
axes[0, 0].legend(fontsize=9, loc='lower right')
set_style(axes[0, 0])

# Grafico 2: Comparacion de Modelos
modelos = ['LightGBM', 'XGBoost', 'RandomForest', 'Ridge']
aucs = [0.8745, 0.8698, 0.8612, 0.8321]

bars = axes[0, 1].barh(modelos, aucs, color=[COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']],
                       edgecolor='black', linewidth=1.5)
axes[0, 1].set_xlabel('AUC', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Comparacion de Modelos', fontsize=12, fontweight='bold', pad=15)
axes[0, 1].set_xlim([0.82, 0.88])
for i, (bar, auc) in enumerate(zip(bars, aucs)):
    axes[0, 1].text(auc - 0.0005, i, f'{auc:.4f}', va='center', ha='right',
                   fontweight='bold', fontsize=9, color='white')
set_style(axes[0, 1])

# Grafico 3: Churn Real vs AUC
semanas_labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
churn_real = [63.5, 67.2, 64.1, 68.9, 65.3, 66.7, 64.8, 67.1]
auc_sem = [0.851, 0.864, 0.859, 0.876, 0.868, 0.872, 0.875, 0.874]

ax2 = axes[1, 0].twinx()
bars = axes[1, 0].bar(semanas_labels, churn_real, color=COLORS['warning'], alpha=0.7, edgecolor='black', linewidth=1.5)
line = ax2.plot(semanas_labels, auc_sem, marker='D', color=COLORS['success'], linewidth=2.5, markersize=8,
               markeredgecolor='black', markeredgewidth=1.5)

axes[1, 0].set_ylabel('Churn Real (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('AUC', fontsize=11, fontweight='bold', color=COLORS['success'])
axes[1, 0].set_title('Churn Real vs AUC por Semana', fontsize=12, fontweight='bold', pad=15)
axes[1, 0].set_ylim([62, 70])
ax2.set_ylim([0.84, 0.88])
set_style(axes[1, 0])

# Grafico 4: Precision vs Cobertura por Canal
canales = ['App Mobile', 'Web', 'Desktop', 'Telefono']
precision_critico = [89.2, 88.5, 91.2, 87.8]
cobertura = [72.1, 71.8, 70.5, 73.2]

x = np.arange(len(canales))
width = 0.35
bars1 = axes[1, 1].bar(x - width/2, precision_critico, width, label='Precision (%)',
                       color=COLORS['success'], edgecolor='black', linewidth=1)
bars2 = axes[1, 1].bar(x + width/2, cobertura, width, label='Cobertura (%)',
                       color=COLORS['primary'], edgecolor='black', linewidth=1)

axes[1, 1].set_ylabel('Porcentaje (%)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Precision vs Cobertura por Canal', fontsize=12, fontweight='bold', pad=15)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(canales, fontsize=9, rotation=15)
axes[1, 1].set_ylim([65, 95])
axes[1, 1].legend(fontsize=9, loc='lower right')
set_style(axes[1, 1])

plt.tight_layout()
plt.savefig('graficos_churn_gaming.png', dpi=300, bbox_inches='tight', facecolor='white')
print("OK - graficos_churn_gaming.png")
