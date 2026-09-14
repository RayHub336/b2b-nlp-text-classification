import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from data_prep import prep_data

def train_and_evaluate():
    print("Iniciando pipeline de entrenamiento NLP B2B...")
    
    # 1. Obtener matrices matemáticas (Aplica la limpieza y vectorización TF-IDF)
    X_train, X_test, y_train, y_test = prep_data()

    # 2. Configuración del Algoritmo Predictivo
    print("\nEntrenando modelo de clasificación de textos...")
    # Utilizamos el modelo logístico extraído de la fase de experimentación
    model = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
    model.fit(X_train, y_train)

    # 3. Auditoría de Calidad (Evals & Métricas)
    print("\nAuditando métricas de rendimiento en lote de validación...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)

    print("-" * 45)
    print("KPIs de Producción Alcanzados:")
    print(f"F1-Score:      {f1:.4f} (Objetivo: > 0.85)")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("-" * 45)

    # 4. Persistencia del Modelo B2B
    joblib.dump(model, 'src/nlp_classification_model.joblib')
    print("\nModelo predictivo guardado exitosamente (.joblib) para consumo automatizado.")

if __name__ == "__main__":
    train_and_evaluate()