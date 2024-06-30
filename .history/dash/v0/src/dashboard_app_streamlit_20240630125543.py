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
            angles = np.arange(0, 360, (360. / len(variables)))

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
    def plot(client, param, genre, organization_type, education_type, income, family, df_group):
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