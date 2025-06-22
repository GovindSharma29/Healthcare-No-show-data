import pandas as pd
df = pd.read_csv("C:/Users/samar/Downloads/KaggleV2-May-2016.csv/KaggleV2-May-2016.csv")
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['AppointmentWeekDay'] = df['AppointmentDay'].dt.day_name()
df['ScheduledWeekDay'] = df['ScheduledDay'].dt.day_name()
df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df = df[df['Age'] >= 0]
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})  # 1 = No-show
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
X = df[['Age', 'SMS_received', 'WaitingDays']]
y = df['No-show']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
df.to_csv("cleaned_no_show_data.csv", index=False)
