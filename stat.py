from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import t

x = norm.ppf(0.25, loc=3, scale=7)
z = norm.ppf(0.25, loc=0, scale=1)

z = norm.rvs(loc=0, scale=1,size=1000)
z

sns.histplot(z, stat="density", color = "gray")

zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()


# X~N(3, sqrt2^2)
# sqrt2*z + 3
x=z*np.sqrt(2)+3

sns.histplot(z, stat="density", color = "gray")
sns.histplot(x, stat="density", color = "gray")

zmin, xmax = (z.min(), x.max())
z_values = np.linspace(zmin, xmax, 100, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.plot(z_values, pdf_values2, color='red', linewidth=2)
plt.show()
plt.clf()



# 표본표준편차 나눠도 표준정규분포가 될까?
x = norm.rvs(loc=5, scale=3, size=10)




#### 자유도가 4인 t분포의 pdf를 그려보세요!
t_values = np.linspace(-4,4,100)
pdf_values = t.pdf(t_values, df=30)
plt.plot(t_values, pdf_values, color="red", linewidth=2)
# 표준 정규분포 겹치기
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color='black', linewidth=2)

plt.show()
plt.clf()


# X~(mu, sigma^2)
# X bar ~ N(mu, sigma^2/n)
# X bar ~ t(x_bar, s^2/n) 자유도가 n-1인 t분포
x = norm.rvs(loc=15, scale=3, size=16, random_state=42)
x
df = len(x)
x_bar=x.mean()

# 모분산을 모를 때: 모평균에 대한 95%에 대한 신뢰구간을 구해보자!
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1)/np.sqrt(df)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1)/np.sqrt(df)

# 모분산을 알 때: 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + norm.ppf(0.975, loc=0, scale=1) *3 / np.sqrt(df)
x_bar - norm.ppf(0.975, loc=0, scale=1) *3 / np.sqrt(df)





























