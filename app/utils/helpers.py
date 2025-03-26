import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(confusion_matrix, classes=['Normal', 'Fraude']):
    """Crée une heatmap de la matrice de confusion avec Plotly."""
    fig = px.imshow(confusion_matrix,
                    labels=dict(x="Prédiction", y="Réalité"),
                    x=classes,
                    y=classes,
                    color_continuous_scale='RdBu_r')
    
    # Ajout des annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            fig.add_annotation(
                x=j,
                y=i,
                text=str(confusion_matrix[i, j]),
                showarrow=False,
                font=dict(color='white' if confusion_matrix[i, j] > confusion_matrix.max()/2 else 'black')
            )
    
    fig.update_layout(title="Matrice de Confusion")
    return fig

def plot_roc_curve(y_true, y_pred_proba):
    """Crée une courbe ROC avec Plotly."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                            name=f'ROC (AUC = {roc_auc:.3f})',
                            mode='lines'))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            name='Random',
                            mode='lines',
                            line=dict(dash='dash')))
    
    fig.update_layout(
        title='Courbe ROC',
        xaxis_title='Taux de faux positifs',
        yaxis_title='Taux de vrais positifs',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700,
        height=500
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """Crée un graphique des importances des features."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None
    
    indices = np.argsort(importances)[::-1]
    
    fig = px.bar(
        x=[feature_names[i] for i in indices[:10]],
        y=importances[indices[:10]],
        title="Top 10 des features les plus importantes"
    )
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Importance",
        xaxis_tickangle=-45
    )
    
    return fig

def format_currency(amount):
    """Formate un montant en euros."""
    return f"{amount:,.2f} €"

def calculate_fraud_statistics(df):
    """Calcule les statistiques de fraude."""
    total_transactions = len(df)
    fraud_transactions = df['Class'].sum()
    fraud_rate = (fraud_transactions / total_transactions) * 100
    
    return {
        'total_transactions': total_transactions,
        'fraud_transactions': int(fraud_transactions),
        'fraud_rate': fraud_rate
    }

def cluster_transactions(df, n_clusters):
    """Crée des clusters de transactions."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    # Sélection des features à utiliser
    features = df.drop(['Time', 'Class'], axis=1)
    
    # Standardisation des features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Réduction de dimension
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_features)
    
    # Créer un DataFrame avec les résultats
    result_df = pd.DataFrame(reduced_features, columns=['x', 'y'])
    result_df['Cluster'] = clusters
    
    return result_df