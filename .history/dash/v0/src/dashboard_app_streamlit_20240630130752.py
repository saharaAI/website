import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


class DataLoader:
    @staticmethod
    def load_data():
        df = pd.read_csv("../data/data_api.csv")
        """df_int = pd.read_csv('df_interprete')
        df_int.drop('Unnamed: 0', axis=1, inplace=True)
        df_group = pd.read_csv('df_group')
        df_int_sans_unite = pd.read_csv('df_interp_sans_unite')
        genre = pd.read_csv('genre')
        income = pd.read_csv('income')
        education_type = pd.read_csv('education_type')
        organization_type = pd.read_csv('organization_type')
        family = pd.read_csv('family')
        df_nn = pd.read_csv('df_nn')"""
        return df #, df_int, df_group, df_int_sans_unite, genre, income, education_type, organization_type, family, df_nn

    @staticmethod
    def load_models():
        with open('LightGBMModel.pkl', 'rb') as f:
            lgbm = pickle.load(f)
        with open('NearestNeighborsModel.pkl', 'rb') as f:
            nn = pickle.load(f)
        with open('StandardScaler.pkl', 'rb') as f:
            std = pickle.load(f)
        return lgbm, nn, std

class RadarChart:
    @staticmethod
    def _invert(x, limits):
        return limits[1] - (x - limits[0])

    @staticmethod
    def _scale_data(data, ranges):
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            assert (y1 <= d <= y2) or (y2 <= d <= y1)
        x1, x2 = ranges[0]
        d = data[0]
        if x1 > x2:
            d = RadarChart._invert(d, (x1, x2))
            x1, x2 = x2, x1
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            if y1 > y2:
                d = RadarChart._invert(d, (y1, y2))
                y1, y2 = y2, y1
            sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)
        return sdata

    class ComplexRadar:
        def __init__(self, fig, variables, ranges, n_ordinate_levels=6):
            angles = np.arange(0, 360, 360. / len(variables))

            axes = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True,
                                 label="axes{}".format(i))
                    for i in range(len(variables))]

            axes[0].set_thetagrids(angles, labels=[])

            for ax in axes[1:]:
                ax.patch.set_visible(False)
                ax.grid("off")
                ax.xaxis.set_visible(False)

            for i, ax in enumerate(axes):
                grid = np.linspace(*ranges[i], num=n_ordinate_levels)
                gridlabel = ["{}".format(round(x, 2)) for x in grid]
                if ranges[i][0] > ranges[i][1]:
                    grid = grid[::-1]
                gridlabel[0] = ""
                ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
                ax.set_ylim(*ranges[i])

            ticks = angles
            ax.set_xticks(np.deg2rad(ticks))
            ticklabels = variables
            ax.set_xticklabels(ticklabels, fontsize=10)

            angles1 = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
            angles1[np.cos(angles1) < 0] = angles1[np.cos(angles1) < 0] + np.pi
            angles1 = np.rad2deg(angles1)
            labels = []
            for label, angle in zip(ax.get_xticklabels(), angles1):
                x, y = label.get_position()
                lab = ax.text(x, y - .5, label.get_text(), transform=label.get_transform(),
                              ha=label.get_ha(), va=label.get_va())
                lab.set_rotation(angle)
                lab.set_fontsize(16)
                lab.set_fontweight('bold')
                labels.append(lab)
            ax.set_xticklabels([])

            self.angle = np.deg2rad(np.r_[angles, angles[0]])
            self.ranges = ranges
            self.ax = axes[0]

        def plot(self, data, *args, **kw):
            sdata = RadarChart._scale_data(data, self.ranges)
            self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

        def fill(self, data, *args, **kw):
            sdata = RadarChart._scale_data(data, self.ranges)
            self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    @staticmethod
    def plot(client, param, genre, organization_type, education_type, income, family, df_group, identifiant):
        def ok_impayes(df, var_name):
            cat = df_group[df_group['SK_ID_CURR'] == identifiant][var_name].iloc[0]
            ok = df[df[var_name] == cat][df['TARGET'] == 0]
            ok.drop(['TARGET', var_name], inplace=True, axis=1)
            impayes = df[df[var_name] == cat][df['TARGET'] == 1]
            impayes.drop(['TARGET', var_name], inplace=True, axis=1)
            return ok, impayes, cat

        if param == 'Genre':
            df = genre.copy()
            ok, impayes, cat = ok_impayes(df, 'CODE_GENDER')
            client = pd.concat([client, ok, impayes], ignore_index=True)
        elif param == "Type d'entreprise":
            df = organization_type.copy()
            ok, impayes, cat = ok_impayes(df, 'ORGANIZATION_TYPE')
            client = pd.concat([client, ok, impayes], ignore_index=True)
        elif param == "Niveau d'éducation":
            df = education_type.copy()
            ok, impayes, cat = ok_impayes(df, 'NAME_EDUCATION_TYPE')
            client = pd.concat([client, ok, impayes], ignore_index=True)
        elif param == "Niveau de revenus":
            df = income.copy()
            ok, impayes, cat = ok_impayes(df, 'AMT_INCOME')
            client = pd.concat([client, ok, impayes], ignore_index=True)
        elif param == 'Statut marital':
            df = family.copy()
            ok, impayes, cat = ok_impayes(df, 'NAME_FAMILY_STATUS')
            client = pd.concat([client, ok, impayes], ignore_index=True)

        variables = ("Durée emprunt", "Annuités", "Âge", "Début contrat travail", "Annuités/revenus")
        data_ex = client.iloc[0]
        ranges = [(min(client["Durée emprunt"]) - 5, max(client["Durée emprunt"]) + 1),
                  (min(client["Annuités"]) - 5000, max(client["Annuités"]) + 5000),
                  (min(client["Âge"]) - 5, max(client["Âge"]) + 5),
                  (min(client["Début contrat travail"]) - 1, max(client["Début contrat travail"]) + 1),
                  (min(client["Annuités/revenus"]) - 5, max(client["Annuités/revenus"]) + 5)]

        fig1 = plt.figure(figsize=(6, 6))
        radar = RadarChart.ComplexRadar(fig1, variables, ranges)
        radar.plot(data_ex, label='Notre client')
        radar.fill(data_ex, alpha=0.2)

        radar.plot(ok.iloc[0], label='Moyenne des clients similaires sans défaut de paiement', color='g')
        radar.plot(impayes.iloc[0], label='Moyenne des clients similaires avec défaut de paiement', color='r')

        fig1.legend(bbox_to_anchor=(1.7, 1))

        st.pyplot(fig1)

class Plotter:
    @staticmethod
    def bar_plot(df, col):
        labels = {
            'CODE_GENDER': 'Genre',
            'ORGANIZATION_TYPE': "Type d'entreprise",
            'NAME_EDUCATION_TYPE': "Niveau d'éducation",
            'AMT_INCOME': "Niveau de revenus",
            'NAME_FAMILY_STATUS': 'Statut marital',
            'Count': 'Effectif'
        }
        titre = f"Répartition du nombre et du pourcentage d'impayés suivant le {str.lower(labels[col])}"
        fig = px.bar(df, x=col, y="Count", color="Cible", text="Percentage",
                     labels=labels, color_discrete_sequence=['#90ee90', '#ff4500'], title=titre)
        st.plotly_chart(fig)

    @staticmethod
    def plot_metrics(metrics_list, lgbm, x_test, y_test):
        y_pred = lgbm.predict(x_test)
        y_pred_proba = lgbm.predict_proba(x_test)[:, 1]

        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)

class CreditApplication:
    def __init__(self):
        self.data_loader = DataLoader()
        self.plotter = Plotter()
        self.radar_chart = RadarChart()
        self.df, self.df_int, self.df_group, self.df_int_sans_unite, self.genre, self.income, self.education_type, self.organization_type, self.family, self.df_nn = self.data_loader.load_data()
        self.lgbm, self.nn, self.std = self.data_loader.load_models()

    def run(self):
        st.set_page_config(layout='wide', page_title="Application d'acceptation de crédit")
        st.write("# Application d'acceptation de crédit")

        col1, col2, col3 = st.columns([5, 1, 10])

        with col1:
            st.write("### Seuil d'acceptation de crédit")
            seuil = st.slider("Choisissez la probabilité d'impayés acceptée :", min_value=0.00, max_value=1.00, value=0.50, step=0.01)
            st.write("### Renseignez le numéro client :")
            identifiant = st.number_input(' ', min_value=100002, max_value=112188)

        if (self.df['SK_ID_CURR'] == identifiant).sum() == 0:
            st.write("Identifiant client inconnu")
        else:
            self.process_client(identifiant, seuil, col1, col3)

    def process_client(self, identifiant, seuil, col1, col3):
        df_client = self.df[self.df['SK_ID_CURR'] == identifiant]
        df_client_int = self.df_int[self.df_int['Identifiant'] == identifiant]
        df_client_int_SU = self.df_int_sans_unite[self.df_int_sans_unite['Identifiant'] == identifiant]
        df_client_int.set_index('Identifiant', inplace=True)

        with col3:
            st.write('## Informations du client', df_client_int.drop('Défaut paiement', axis=1))

        with col1:
            self.display_prediction(df_client, identifiant, seuil, df_client_int)

        self.display_nearest_neighbors(df_client, col3)
        self.display_model_metrics()
        self.display_comparison_graphs(df_client_int_SU, identifiant)

    def display_prediction(self, df_client, identifiant, seuil, df_client_int):
        feats = [f for f in df_client.columns if f not in ['SK_ID_CURR', 'TARGET']]
        results = pd.DataFrame(self.lgbm.predict_proba(df_client[feats]), index=[identifiant])
        results.rename({0: "Absence de défaut de paiement", 1: "Probabilité d'impayés"}, axis=1, inplace=True)
        st.write("## Prédiction", results["Probabilité d'impayés"])
        
        proba = results["Probabilité d'impayés"].iloc[0]
        def_p = "Le client a déjà été en défaut de paiement : " + str(df_client_int['Défaut paiement'].iloc[0])
        
        if proba < seuil:
            st.markdown("Résultat : :white_check_mark: **Client présente un faible risque d'impayés** ")
        else:
            st.write("Résultat : :warning: **Risque d'impayés important, client à surveiller** ")
        st.write('>', def_p)
        st.write(" ")

 
    def display_nearest_neighbors(self, df_client, col3):
        interpretable_important_data = ['SK_ID_CURR', 'PAYMENT_RATE', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'ANNUITY_INCOME_PERC']
        client_list = self.std.transform(df_client[interpretable_important_data])
        distance, voisins = self.nn.kneighbors(client_list)
        voisins = voisins[0]
        voisins_table = pd.DataFrame({v: self.df_nn.iloc[voisins[v]] for v in range(len(voisins))})
        
        with col3:
            st.write("## Profils de clients similaires en base")
            voisins_int = pd.DataFrame(index=range(len(voisins_table.transpose())), columns=self.df_int.columns)
            for i, id in enumerate(voisins_table.transpose()['SK_ID_CURR']):
                voisins_int.iloc[i] = self.df_int[self.df_int['Identifiant'] == id]
            voisins_int.set_index('Identifiant', inplace=True)
            st.write(voisins_int)

    def display_model_metrics(self):
        st.write("## Métriques d'entrainement du modèle ")
        with st.expander("Afficher les métriques"):
            col1_1, col2_1, col3_1 = st.columns([3, 1, 5])
            with col1_1:
                metrics = st.selectbox(label=" Choisissez la métrique à voir : ",
                                       options=('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
            with col3_1:
                x_test = self.df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
                y_test = self.df['TARGET']
                self.plotter.plot_metrics([metrics], self.lgbm, x_test, y_test)

    def display_comparison_graphs(self, df_client_int_SU, identifiant):
        st.write("## Graphiques interactifs de comparaison entre le client et un groupe d'individus similaires")
        with st.expander("Afficher les graphiques"):
            col1_2, col2_2, col3_2 = st.columns([10, 1, 10])
            with col1_2:
                param = st.selectbox(label=" Choisissez le paramètre à comparer : ",
                                     options=('Genre', "Type d'entreprise", "Niveau d'éducation", "Niveau de revenus", "Statut marital"))
                self.display_category_info(param, identifiant)
            with col3_2:
                st.write(f"Graphe radar comparant notre client aux clients du même {str.lower(param)}")
                self.radar_chart.plot(df_client_int_SU.drop('Identifiant', axis=1), param, self.genre, self.organization_type, self.education_type, self.income, self.family, self.df_group, identifiant)

    def display_category_info(self, param, identifiant):
        if param == 'Genre':
            cat = self.df_group[self.df_group['SK_ID_CURR'] == identifiant]['CODE_GENDER'].iloc[0]
            st.write('Le client est une femme.' if cat == 'F' else 'Le client est un homme.')
            self.plotter.bar_plot(self.genre, 'CODE_GENDER')
        elif param == "Type d'entreprise":
            cat = self.df_group[self.df_group['SK_ID_CURR'] == identifiant]['ORGANIZATION_TYPE'].iloc[0]
            st.write("Type d'entreprise du client : " + cat)
            self.plotter.bar_plot(self.organization_type, 'ORGANIZATION_TYPE')
        elif param == "Niveau d'éducation":
            cat = self.df_group[self.df_group['SK_ID_CURR'] == identifiant]['NAME_EDUCATION_TYPE'].iloc[0]
            st.write("Niveau d'éducation du client : " + cat)
            self.plotter.bar_plot(self.education_type, 'NAME_EDUCATION_TYPE')
        elif param == "Niveau de revenus":
            cat = self.df_group[self.df_group['SK_ID_CURR'] == identifiant]['AMT_INCOME'].iloc[0]
            st.write("Le client a des revenus situés " + cat)
            self.plotter.bar_plot(self.income, 'AMT_INCOME')
        elif param == "Statut marital":
            cat = self.df_group[self.df_group['SK_ID_CURR'] == identifiant]['NAME_FAMILY_STATUS'].iloc[0]
            st.write("Statut marital du client : " + cat)
            self.plotter.bar_plot(self.family, 'NAME_FAMILY_STATUS')

if __name__ == "__main__":
    app = CreditApplication()
    app.run()