import pandas as pd
df = pd.read_csv('data/titanic.csv')
#print(df.shape); print(df.head()); df.info(); print(df.describe())

kids = df[df['age'] < 12]
#print(len(kids), kids['survived'].mean())

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['deck'])
#print(df.isna().sum())

# print(df.groupby('sex')['survived'].mean())
# print(df.groupby('pclass')['survived'].mean())
# print(df.groupby(['pclass', 'sex'])['age'].mean().round(1))

import matplotlib.pyplot as plt, seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
ax.hist(df[df['survived'] == 0]['age'], bins=30, alpha=0.5, label='Погибшие'); 
ax.hist(df[df['survived'] == 1]['age'], bins=30, alpha=0.5, label='Выжившие'); ax.set_title('Возраст'); ax.legend()

# sns.boxplot(x='pclass', y='fare', data=df, ax=ax[1]); ax[1].set_title('Цена по классу')
# sns.barplot(x='sex', y='survived', data=df, ax=ax[2]); ax[2].set_title('Выживаемость')
fig.tight_layout(); fig.savefig('titanic.png', dpi=120)
plt.show()

df['family'] = df['sibsp'] + df['parch']
df['sex_num'] = df['sex'].map({'female': 1, 'male': 0})
df.to_csv('data/clean.csv', index=False)

cf = pd.read_csv('data/clean.csv')
print(cf.isna().sum())

