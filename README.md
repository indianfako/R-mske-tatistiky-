# R-mske-tatistiky-
Johan fako 1986
# importujeme potrebné knižnice
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # pridáme knižnicu seaborn
import corrplot # pridáme knižnicu corrplot

# vytvoríme náhodné dáta s 100 pozorovaniami a 4 premennými
np.random.seed(42) # nastavíme náhodný zdroj pre reprodukovateľnosť
data = np.random.randn(100, 4) # generujeme náhodné normálne rozdelené hodnoty
df = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4"]) # vytvoríme dátový rámec s pandas

# vypočítame kovariančnú maticu, ktorá obsahuje kovariancie medzi každou dvojicou premenných
cov_matrix = np.cov(data, rowvar=False) # použijeme funkciu np.cov s parametrom rowvar=False
print("Kovariančná matica:")
print(cov_matrix)

# vypočítame korelačnú maticu, ktorá obsahuje Pearsonove korelačné koeficienty medzi každou dvojicou premenných
# Pearsonov korelačný koeficient meria lineárnu závislosť medzi premennými
cor_matrix = np.corrcoef(data, rowvar=False) # použijeme funkciu np.corrcoef s parametrom rowvar=False
print("Korelačná matica:")
print(cor_matrix)

# vypočítame Spearmanove a Kendallove korelačné koeficienty, ktoré merajú monotónnu závislosť medzi premennými
# Spearmanov korelačný koeficient je založený na poradí hodnôt, nie na ich skutočných hodnotách
# Kendallov korelačný koeficient je založený na počte zhodných a nezhodných párov hodnôt
spearman_matrix = scipy.stats.spearmanr(data) # použijeme funkciu scipy.stats.spearmanr
kendall_matrix = scipy.stats.kendalltau(data) # použijeme funkciu scipy.stats.kendalltau
print("Spearmanove korelačné koeficienty:")
print(spearman_matrix.correlation) # vypíšeme iba maticu korelácií, nie p-hodnoty
print("Kendallove korelačné koeficienty:")
print(kendall_matrix.correlation) # vypíšeme iba maticu korelácií, nie p-hodnoty

# vizualizujeme korelačnú maticu pomocou heatmapy s matplotlib
plt.imshow(cor_matrix, cmap="coolwarm") # použijeme funkciu plt.imshow s farebnou mapou coolwarm
plt.colorbar() # pridáme farebnú stupnicu
plt.xticks(np.arange(4), ["x1", "x2", "x3", "x4"]) # nastavíme popisky osi x
plt.yticks(np.arange(4), ["x1", "x2", "x3", "x4"]) # nastavíme popisky osi y
plt.title("Korelačná matica") # pridáme nadpis
plt.show() # zobrazíme graf

# vizualizujeme korelačnú maticu pomocou funkcie corrplot z knižnice corrplot
corrplot.corrplot(cor_matrix, method="color") # použijeme funkciu corrplot.corrplot s parametrom method="color"
plt.show() # zobrazíme graf

# vizualizujeme párové grafy medzi všetkými premennými pomocou funkcie pairplot z knižnice seaborn
sns.pairplot(df) # použijeme funkciu sns.pairplot s dátovým rámcom df
plt.show() # zobrazíme graf

# získame základné štatistiky o dátach pomocou funkcie describe z knižnice pandas
print("Základné štatistiky:")
print(df.describe()) # použijeme funkciu df.describe
