from sklearn.ensemble import RandomForestClassifier

def train_dummy_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    return model

def predict_uc_probability(model, X):
    # Dummy trained logic (replace with real labeled UC data later)
    prob = model.predict_proba(X)[0][1]
    return round(prob * 100, 2)
