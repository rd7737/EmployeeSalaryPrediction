from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import joblib

def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Regressor": SVR()
    }

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = get_models()
    scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = {
            'model': model,
            'r2_score': r2_score(y_test, y_pred)
        }
        
    return scores

def save_best_model(models_scores, path='models/trained_model.pkl'):
    best_model = max(models_scores.items(), key=lambda x: x[1]['r2_score'])[1]['model']
    joblib.dump(best_model, path)
