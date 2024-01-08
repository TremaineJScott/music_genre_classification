import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

class GenreClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()

    def load_data(self):
        self.data = pd.read_csv(self.dataset_path)

    def preprocess_data(self):
        # Selecting the new subset of features
        feature_columns = ['key', 'mode', 'tempo']
        X = self.data[feature_columns]
        y = self.data['music_genre']

        # Encode 'key' and 'mode' columns
        self.encoder = LabelEncoder()
        X['key'] = self.encoder.fit_transform(X['key'])
        X['mode'] = self.encoder.fit_transform(X['mode'])

        # Scaling 'tempo' feature
        scaler = StandardScaler()
        X[['tempo']] = scaler.fit_transform(X[['tempo']])
        
        self.X_scaled = X
        self.y = y

    def train_model(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        self.svm_classifier = SVC(kernel='linear')
        self.svm_classifier.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.svm_classifier.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(f'Accuracy: {accuracy_score(self.y_test, y_pred)}')

    def predict(self, features):
        # Scaling 'tempo' for the incoming features and encoding 'key' and 'mode'
        features = list(features)
        features[0] = self.encoder.transform([features[0]])[0]  # 'key'
        features[1] = self.encoder.transform([features[1]])[0]  # 'mode'
        features[2] = (features[2] - self.data['tempo'].mean()) / self.data['tempo'].std()  # 'tempo'
        return self.svm_classifier.predict([features])
