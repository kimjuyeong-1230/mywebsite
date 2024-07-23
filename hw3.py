import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

## 1번
# 데이터 불러오기
mpg = pd.read_csv("data/mpg.csv")
mpg

# 그래프 그리기
sns.scatterplot(data= mpg, x='cty', y='hwy')
plt.show()
plt.clf()


## 2번
# 데이터 불러오기 
midwest = pd.read_csv("data/midwest.csv")
midwest

# 그래프 그리기
sns.scatterplot(data= midwest, x='poptotal', y='popasian') \
  .set(xlim =[0,500000], ylin=[0,10000])
plt.show()



#pg.211
## 1번
# 데이터 불러오기
mpg = pd.read_csv("data/mpg.csv")
mpg

df = mpg.query('category == "suv"') \
        .groupby('manufacturer', as_index=False) \
        .agg(mean_cty = ('cty', 'mean')) \
        .sort_values('mean_cty', ascending = False) \
        .head()
df

# 막대그래프
sns.barplot(data = df, x = 'manufacturer', y = 'mean_cty')
plt.show()
plt.clf()


## 2번
df_mpg = mpg.groupby('category', as_index=False) \
        .agg(n = ('category', 'count')) \
        .sort_values('n', ascending = False)
df_mpg

# 막대그래프
sns.barplot(data = df_mpg, x = 'category', y = 'n')
plt.show()





