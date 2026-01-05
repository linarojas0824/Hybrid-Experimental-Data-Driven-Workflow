from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        
    def split_training(self,X,y):
        X_train, X_test, y_train,y_test = train_test_split(X,y,
                                                           test_size=self.test_size,
                                                           random_state=self.random_state,
                                                           shuffle=True)
        return (X_train, X_test, y_train,y_test)

    def fit_transform(self,X_train):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        return X_train_scaled
