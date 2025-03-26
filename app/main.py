import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import numpy as np
from preprocessing.data_processor import DataProcessor
from models.model_training import ModelTrainer
from utils.helpers import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, calculate_fraud_statistics, cluster_transactions

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection de Fdraude Bancaire",
    page_icon="🏦",
    layout="wide"
)

def main():
    st.title("🏦 Système de Détection de Fraude Bancaire")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choisissez une page",
        ["Accueil ", "Analyse des Données", "Entraînement du Modèle", "Prédiction"]
    )
    
    if page == "Accueil":
        show_home()
    elif page == "Analyse des Données":
        show_data_analysis()
    elif page == "Entraînement du Modèle":
        show_model_training()
    elif page == "Prédiction":
        show_prediction()

def show_home():
    st.header("Bienvenue dans l'application de détection de fraude")
    st.write("""
    Cette application vous permet de :
    - Analyser les données bancaires
    - Entraîner des modèles de détection de fraude
    - Effectuer des prédictions sur de nouvelles transactions
    """)
    
    st.info("👈 Utilisez le menu de gauche pour naviguer dans l'application")

def show_data_analysis():
    st.header("Analyse des Données")
    
    try:
        df = pd.read_csv('./data/creditcard.csv')
        st.write("Aperçu des données :")
        st.dataframe(df.head())
        
        # Statistiques de base
        st.subheader("Statistiques descriptives")
        st.write(df.describe())
        
        # Distribution des classes
        st.subheader("Distribution des transactions frauduleuses vs normales")
        stats = calculate_fraud_statistics(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total des transactions", f"{stats['total_transactions']:,}")
        with col2:
            st.metric("Transactions frauduleuses", f"{stats['fraud_transactions']:,}")
        with col3:
            st.metric("Taux de fraude", f"{stats['fraud_rate']:.2f}%")
        
        fig = px.pie(names=['Normal', 'Fraude'], 
                    values=df['Class'].value_counts().values,
                    title="Distribution des classes")
        st.plotly_chart(fig)
        st.subheader("Clustering des transactions")
        n_clusters = st.slider("Nombre de clusters", 2, 10, 3)
        
        if st.button("Créer les clusters"):
            st.info("Création des clusters...")
    
             # Appel de la fonction
            clustered_data = cluster_transactions(df, n_clusters)
    
            fig_clusters = px.scatter(
                clustered_data, 
                x='x',  # Colonne pour l'axe x
                y='y',  # Colonne pour l'axe y
                color='Cluster',
                title="Clustering des transactions"
    )       
    
            st.plotly_chart(fig_clusters)
    except FileNotFoundError:
        st.error("Le fichier de données n'a pas été trouvé. Veuillez placer le fichier creditcard.csv dans le dossier data/")

def show_model_training():
    st.header("Entraînement du Modèle")
    
    try:
        # Chargement des données
        df = pd.read_csv('./data/creditcard.csv')
        
        # Initialisation des processeurs
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        
        # Paramètres du modèle
        st.subheader("Paramètres du modèle Random Forest")
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Nombre d'arbres", 50, 200, 100)
            max_depth = st.slider("Profondeur maximale", 5, 30, 10)
        with col2:
            test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
            use_smote = st.checkbox("Utiliser SMOTE pour le rééquilibrage", value=True)
        
        if st.button("Entraîner le modèle"):
            with st.spinner("Prétraitement des données..."):
                # Préparation des données
                X = df.drop('Class', axis=1)
                y = df['Class']
                
                # Split des données
                X_train, X_test, y_train, y_test = data_processor.split_data(X, y, test_size=test_size)
                
                if use_smote:
                    st.info("Application de SMOTE pour rééquilibrer les données...")
                    X_train, y_train = data_processor.apply_smote(X_train, y_train)
                
                # Configuration et entraînement du modèle
                st.info("Entraînement du modèle Random Forest...")
                model = model_trainer.train_model(
                    X_train, 
                    y_train,
                    model_params={
                        'n_estimators': n_estimators,
                        'max_depth': max_depth
                    }
                )
                
                # Évaluation du modèle
                st.success("Entraînement terminé ! Voici les résultats :")
                
                results = model_trainer.evaluate_model(X_test, y_test)
                
                # Affichage des métriques
                st.subheader("Métriques de performance")
                st.text(results['classification_report'])
                
                # Matrice de confusion
                st.subheader("Matrice de confusion")
                fig_cm = plot_confusion_matrix(results['confusion_matrix'])
                st.plotly_chart(fig_cm)
                
                # Courbe ROC
                st.subheader("Courbe ROC")
                y_pred_proba = model_trainer.predict_proba(X_test)
                fig_roc = plot_roc_curve(y_test, y_pred_proba)
                st.plotly_chart(fig_roc)
                
                # Importance des features
                st.subheader("Importance des features")
                fig_imp = plot_feature_importance(model_trainer.current_model, X.columns)
                if fig_imp is not None:
                    st.plotly_chart(fig_imp)
                
                # Sauvegarde du modèle
                model_trainer.save_model('models/random_forest_model.joblib')
                st.success("Le modèle a été sauvegardé avec succès !")
                
    except Exception as e:
        st.error(f"Une erreur s'est produite : {str(e)}")

def show_prediction():
    st.header("Prédiction de Fraude")
    st.write("Cette section permettra de faire des prédictions sur de nouvelles transactions.")
    
    try:
        model_trainer = ModelTrainer()
        model_trainer.load_model('models/random_forest_model.joblib')
        
        st.info("Entrez les valeurs pour faire une prédiction")
        
        # Création d'un formulaire pour les features
        input_data = {}
        
        col1, col2 = st.columns(2)
        with col1:
            for i in range(0, 15):
                input_data[f'V{i}'] = st.number_input(f'Feature V{i}', value=0.0)
        with col2:
            for i in range(15, 29):
                input_data[f'V{i}'] = st.number_input(f'Feature V{i}', value=0.0)
        
        amount = st.number_input('Montant de la transaction', value=0.0)
        input_data['Amount'] = amount
        
        if st.button('Faire une prédiction'):
            # Préparation des données
            data_processor = DataProcessor()
            input_df = pd.DataFrame([input_data])
            processed_data = data_processor.prepare_single_prediction(input_df)
            
            # Prédiction
            prediction = model_trainer.predict(processed_data)
            probability = model_trainer.predict_proba(processed_data)[0]
            
            # Affichage du résultat
            st.subheader('Résultat de la prédiction')
            if prediction[0] == 1:
                st.error(f'⚠️ Transaction potentiellement frauduleuse (Probabilité: {probability:.2%})')
            else:
                st.success(f'✅ Transaction probablement légitime (Probabilité: {1-probability:.2%})')
                
    except FileNotFoundError:
        st.error("Le modèle n'a pas été trouvé. Veuillez d'abord entraîner un modèle.")
    except Exception as e:
        st.error(f"Une erreur s'est produite : {str(e)}")

if __name__ == "__main__":
    main() 