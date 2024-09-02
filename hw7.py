import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import norm

### ADP 표본점수
# 2022년에 실시 된 ADP 실기 시험의 통계파트 표준점수는 평균이 30, 표준편차가 5인 정규분포를 따른다고 한다.

# 1) ADP 실기 시험의 통계파트 표준점수의 밀도함수를 그려보세요.
x = np.arange(15, 45, 0.1)
y = norm.pdf(x, loc=30,scale=5)
plt.plot(x, y, color='k')
plt.show()

# 2) ADP 수험생을 임의로 1명을 선택하여 통계 점수를 조회했을때 45점 보다 높은 점수를 받았을 확률을 구하세요.
1 - norm.cdf(45, 30, 5)


# 3) 슬통이는 상위 10%에 해당하는 점수를 얻었다고 한다면, 슬통이의 점수는 얼마인지 계산해보세요.
norm.ppf(0.90, loc=30, scale=5)

# 4) 슬기로운 통계생활의 해당 회차 수강생은 16명이었다고 한다. 16명의 통계 파트 점수를 평균내었을 때, 이 평균값이 따르는 분포의 확률밀도 함수를 1번의 그래프와 겹쳐 그려보세요.
x = np.arange(15, 45, 0.1)
y = norm.pdf(x, loc=30,scale=5)
plt.plot(x, y, color='k') # 1번 그래프

y_2 = norm.pdf(x, loc=30,scale=5/np.sqrt(16))
plt.plot(x, y_2, color='red')
plt.show()

# 5) 슬기로운 통계생활 ADP 반 수강생들의 통계점수를 평균내었다고 할 때, 이 값이 38점보다 높게 나올 확률을 구하세요.
1 - norm.cdf(38, loc=30, scale=5/np.sqrt(16))



## Covid 19 발병률
# Covid‑19의 발병률은 1%라고 한다. 다음은 이번 코로나 사태로 인하여 코로나 의심 환자들 1,085명을 대상으로 슬통 회사의 “다잡아” 키트를 사용하여 양성 반응을 체크한 결과이다.

TP = 370  
FN = 15   
FP = 10   
TN = 690  

# 1) 다잡아 키트가 코로나 바이러스에 걸린 사람을 양성으로 잡아낼 확률을 계산하세요.
TP / (TP+FN)

# 2) 슬통 회사에서 다잡아 키트를 사용해 양성으로 나온 사람이 실제로는 코로나 바이러스에 걸려있을 확률을 97%라며, 키트의 우수성을 주장했다. 이 주장이 옳지 않은 이유를 서술하세요.
표본에서의 결과는 모집단과 차이가 있기 때문이다.

# 3) Covid‑19 발병률을 사용하여, 키트의 결과값이 양성으로 나온 사람이 실제로 코로나 바이러스에 걸려있을 확률을 구하세요.
(TP/(TP+FN) * 0.01) / (0.01 * (TP/(TP+FN)) + 0.99 * (FP/(FP+TN)))

## 카이제곱분포와 표본분
# 자유도가 𝑘인 카이제곱분포를 따르는 확률변수 𝑋를 𝑋 ∼ 𝜒2(𝑘)과 같이 나타내고, 이 확률변수의 확률밀도함수는 다음과 같습니다
from scipy.stats import chi2

# 1) 자유도가 4인 카이제곱분포의 확률밀도함수를 그려보세요.
x = np.linspace(0, 20, 1000)
pdf = chi2.pdf(x, 4)
plt.plot(x, pdf)

# 2) 다음의 확률을 구해보세요. 𝑃 (3 ≤ 𝑋 ≤ 5)
chi2.cdf(5, 4) - chi2.cdf(3, 4)

# 3) 자유도가 4인 카이제곱분포에서 크기가 1000인 표본을 뽑은 후, 히스토그램을 그려보세요.
np.random.seed(2024)
sample_data = chi2.rvs(4, size=1000)

plt.hist(sample_data, bins=50, density=True, edgecolor="black")

# 4) 자유도가 4인 카이제곱분포를 따르는 확률변수에서 나올 수 있는 값 중 상위 5%에 해당하는 값은 얼마인지 계산해보세요.
chi2.ppf(0.95, 4)

# 5) 3번에서 뽑힌 표본값들 중 상위 5%에 위치한 표본의 값은 얼마인가요?
np.percentile(sample_data, 95)


# 6) 평균이 3, 표준편차가 2인 정규분포를 따르는 확률변수에서 크기가 20인 표본 𝑥1, ..., 𝑥20,을 뽑은 후 표본분산을 계산한 것을 𝑠2/1이라 생각해보죠. 다음을 수행해보세요!
np.random.seed(2024)
n = 20
var_samples = []

for i in range(500):
    x = norm.rvs(3, 2, size=n)
    var_samples.append(np.var(x, ddof=1))
    
scaled_var_samples = np.array(var_samples) * 4.75

plt.hist(scaled_var_samples,
         bins=50,
         density=True,
         edgecolor="black")

plt.xlabel("4.75 * s^2")
plt.ylabel("density")

x = np.linspace(0, max(scaled_var_samples), 1000)
pdf_chi19 = chi2.pdf(x, df=19)
plt.plot(x, pdf_chi19, 'r--', linewidth=2);
plt.legend(["histogram", "df 19 chisquare dist"], loc="upper right");
plt.show()
