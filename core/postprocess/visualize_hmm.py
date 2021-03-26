import math
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.common.constants import *
from pathlib import Path

FONT_SIZE = 24
FONT_SIZE_SMALL = 12
DPI = 300


class HmmVisualizer:
    @staticmethod
    def save_bar_plot(
            model_filename,
            df_data,
            data_name,
            prob_name,
            legend=True,
            figsize=(9, 6),
            subplots=False,
            layout=None,
    ):
        plt.figure()
        plt.rcParams.update({'font.size': FONT_SIZE_SMALL})
        if legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=2)
        if layout:
            df_data.plot.bar(figsize=figsize, fontsize=FONT_SIZE_SMALL, subplots=subplots, layout=layout, legend=legend, sharex=False, sharey=True, color='black')
        else:
            df_data.plot.bar(figsize=figsize, fontsize=FONT_SIZE_SMALL, subplots=subplots, legend=legend, sharex=False, sharey=True, color='black')
        plt.savefig('{}-{}-{}.png'.format(model_filename, prob_name, data_name), bbox_inches='tight', dpi=DPI)
        plt.close('all')

    @staticmethod
    def save_bar_plot_top(
            model_filename,
            df_data,
            instance_label,
            eval_scale,
            top=3,
            figsize=(6, 4),
            subplots=False,
            drop_special_indices=True
    ):
        df_data = df_data.rename(index={UNK_STR: "Others"})
        if drop_special_indices:
            pass
        value_max = (df_data.max()).max()
        x_max = int(math.ceil(value_max / 100.0)) * 100
        sp_rows = max(int(math.ceil(len(df_data.columns) / 4.0)), 2)
        sp_cols = 4
        fig, axes = plt.subplots(nrows=sp_rows, ncols=sp_cols)
        for state in df_data.columns:
            state_index = int(state[1:])
            sp_row = int(state_index / sp_cols)
            sp_col = state_index - (sp_row * sp_cols)
            axes[sp_row, sp_col].set_xlim(0, x_max)
            axes[sp_row, sp_col].set_title('state{}'.format(state_index))
            axes[sp_row, sp_col].title.set_size(FONT_SIZE)
            state_top = df_data[state].sort_values(ascending=False)[:top][::-1]
            state_top.plot.barh(
                ax=axes[sp_row, sp_col],
                figsize=figsize,
                subplots=subplots,
                colormap='gray',
                fontsize=FONT_SIZE
            )
        filename = Path('{}-top{}-{}-{}.png'.format(model_filename, top, instance_label, eval_scale))
        plt.savefig(filename, bbox_inches='tight', dpi=DPI)
        plt.close('all')

    @staticmethod
    def save_emission_prob_top(
            model,
            model_filename,
            instance_label,
            eval_scale,
            top=3,
            figsize=(6, 4),
            subplots=False,
            drop_special_indices=True
    ):
        state_embeddings = model.state_embeddings
        observation_encodings = model.observation_encodings
        index = ['s{}'.format(i) for i in range(model.num_state)]
        columns = ['{}'.format(model.vocab.i2c[i]) for i in range(len(model.vocab.i2c) - 1)]
        df_data = pd.DataFrame(
            model.emission_net.emission_matrix(state_embeddings, observation_encodings).detach().tolist(),
            index=index,
            columns=columns
        )
        df_data = df_data.T
        df_data = df_data.rename(index={UNK_STR: "Others"})
        if drop_special_indices:
            pass
        x_max = 1.0
        sp_rows = max(int(math.ceil(len(df_data.columns) / 4.0)), 2)
        sp_cols = 4
        fig, axes = plt.subplots(nrows=sp_rows, ncols=sp_cols)
        for state in df_data.columns:
            state_index = int(state[1:])
            sp_row = int(state_index / sp_cols)
            sp_col = state_index - (sp_row * sp_cols)
            axes[sp_row, sp_col].set_xlim(0, x_max)
            axes[sp_row, sp_col].set_title('state{}'.format(state_index))
            axes[sp_row, sp_col].title.set_size(FONT_SIZE)
            state_top = df_data[state].sort_values(ascending=False)[:top][::-1]
            state_top.plot.barh(
                ax=axes[sp_row, sp_col],
                figsize=figsize,
                subplots=subplots,
                colormap='gray',
                fontsize=FONT_SIZE
            )
        filename = Path('{}-emission-prob-top{}-{}-{}.png'.format(model_filename, top, instance_label, eval_scale))
        plt.savefig(filename, bbox_inches='tight', dpi=DPI)
        plt.close('all')

    @staticmethod
    def save_residence_prob_context_beat_4_4(
            model,
            model_filename,
            instance_label,
            eval_scale,
            figsize=(6, 4)
    ):
        state_embeddings = model.state_embeddings
        model.residence_net.batch_size = 1
        beats = [
            1.0, 1.25, 1.5, 1.75,
            2.0, 2.25, 2.5, 2.75,
            3.0, 3.25, 3.5, 3.75,
            4.0, 4.25, 4.5, 4.75
        ]
        sp_rows = max(int(math.ceil(model.num_state / 4.0)), 2)
        sp_cols = 4
        fig, axes = plt.subplots(nrows=sp_rows, ncols=sp_cols, figsize=figsize)
        max_weight = 1.0
        all_residence_probs = []
        for state in range(model.num_state):
            sp_row = int(state / sp_cols)
            sp_col = state - (sp_row * sp_cols)
            axes[sp_row, sp_col].set_title('state{}'.format(state))
            axes[sp_row, sp_col].title.set_size(FONT_SIZE/2)
            axes[sp_row, sp_col].set_aspect('equal', 'box')
            axes[sp_row, sp_col].xaxis.set_major_locator(plt.NullLocator())
            axes[sp_row, sp_col].yaxis.set_major_locator(plt.NullLocator())
            residence_probs = []
            for beat in beats:
                if model.use_beat:
                    beat_tensor = torch.tensor([beat]).to(model.device)
                    residence_context = model.beat_encoder(timesig_numerator=torch.tensor([4.0]).to(model.device),
                                                           beat=beat_tensor).to(model.device)
                else:
                    residence_context = None
                rp = model.residence_net(state_embeddings, context=residence_context)
                rp = rp[:, state, :].squeeze()
                residence_probs.append(rp.tolist())
            for (x, y), w in np.ndenumerate(np.array(residence_probs)):
                color = 'black'
                size = np.sqrt(np.abs(w) / max_weight)
                rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color)
                axes[sp_row, sp_col].add_patch(rect)
            axes[sp_row, sp_col].autoscale_view()
            axes[sp_row, sp_col].invert_yaxis()
            all_residence_probs.append(residence_probs)
        if model.num_state <= 4:
            for dummy_state in range(model.num_state, 8):
                sp_row = int(dummy_state / sp_cols)
                sp_col = dummy_state - (sp_row * sp_cols)
                axes[sp_row, sp_col].set_aspect('equal', 'box')
                axes[sp_row, sp_col].xaxis.set_major_locator(plt.NullLocator())
                axes[sp_row, sp_col].yaxis.set_major_locator(plt.NullLocator())
        filename = Path('{}-residence-beat4-4-prob-{}-{}.png'.format(model_filename, instance_label, eval_scale))
        plt.savefig(filename, bbox_inches='tight', dpi=DPI)
        plt.close('all')
