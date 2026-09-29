import matplotlib
matplotlib.use('Agg')
# =====================================================================
# Load both saved models and the shared preprocessing
# =====================================================================
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tensorflow import keras
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'artifacts')   # models + preprocessing exported from Colab
TEST_DIR = os.path.join(HERE, 'samples')    # example test transformers for the demo tab
BASE_PATH = TEST_DIR                        # the label files sit next to the samples
sys.path.insert(0, OUT_DIR)
import tx_preprocessing as txp
STEP_TO_DAYS = 0.5  # 1 reading every 12 hours

FDD_CLASS_NAMES = {1: 'Normal', 2: 'Partial discharge', 3: 'Low-energy discharge',
                   4: 'Low-temperature overheating'}

# Shared preprocessing: validation rules + gas scaler fitted on the training data
app_pre = txp.TransformerPreprocessor.load(os.path.join(OUT_DIR, 'preprocessor.json'))

# FDD model (PyTorch) - same architecture as the FDD training section
class GRUClassifier(nn.Module):
    def __init__(self, n_features=4, hidden_size=64, num_layers=1, n_classes=4, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_size // 2, n_classes))

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1])

fdd_bundle = torch.load(os.path.join(OUT_DIR, 'fdd_gru_model.pt'), map_location='cpu')
app_fdd_model = GRUClassifier(**fdd_bundle['model_config'])
app_fdd_model.load_state_dict(fdd_bundle['model_state'])
app_fdd_model.eval()
FDD_INDEX_TO_CLASS = {int(k): int(v) for k, v in fdd_bundle['index_to_fdd'].items()}
FDD_THRESHOLD = float(fdd_bundle['confidence_handling']['low_confidence_threshold'])

# RUL model (Keras) - scaler and error range saved in the RUL handoff section, in days
app_rul_model = keras.models.load_model(os.path.join(OUT_DIR, 'rul_cnn_gru_model.keras'))
with open(os.path.join(OUT_DIR, 'rul_target_scaler.json')) as f:
    app_rul_scaler = json.load(f)
with open(os.path.join(OUT_DIR, 'rul_deployment_metadata.json')) as f:
    app_rul_meta = json.load(f)
RUL_RADIUS = float(app_rul_meta['uncertainty_metadata']['error_radius_days'])
RUL_MIN_DAYS = float(app_rul_scaler['data_min_'][0])
RUL_MAX_DAYS = float(app_rul_scaler['data_max_'][0])

# True answers for the test transformers, shown in the demo tab only
def _load_labels(name, scale=1.0):
    path = os.path.join(BASE_PATH, name)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    col = [c for c in df.columns if c != 'id'][0]
    return dict(zip(df['id'], df[col] * scale))

TRUE_FDD = _load_labels('labels_fdd_test.csv')
TRUE_RUL_DAYS = _load_labels('labels_rul_test.csv', STEP_TO_DAYS)
TEST_FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TEST_DIR, '*.csv'))
                    if not os.path.basename(p).startswith('labels_'))

print(f'FDD model loaded | low-confidence threshold {FDD_THRESHOLD * 100:.1f}%')
print(f'RUL model loaded | range {RUL_MIN_DAYS:.1f}-{RUL_MAX_DAYS:.1f} days | error range +/-{RUL_RADIUS:.1f} days')
print(f'{len(TEST_FILES)} test transformers available for the demo')


# =====================================================================
# Prediction: one transformer's readings -> FDD + RUL report and chart
# =====================================================================
def analyse(raw):
    """raw: validated (420, 4) array of gas readings -> dict of results."""
    x = app_pre.transform(raw)[None]  # (1, 420, 4), scaled exactly like the training data

    with torch.no_grad():
        probs = torch.softmax(app_fdd_model(torch.from_numpy(x).float()), dim=1)[0].numpy()
    idx = int(probs.argmax())

    scaled = float(app_rul_model.predict(x, verbose=0).flatten()[0])
    rul_days = max(0.0, (scaled - app_rul_scaler['min_'][0]) / app_rul_scaler['scale_'][0])

    return {'fdd_class': FDD_INDEX_TO_CLASS[idx], 'confidence': float(probs[idx]),
            'probs': {FDD_INDEX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)},
            'rul_days': rul_days}

def build_report(r, true_fdd=None, true_rul=None):
    cls = r['fdd_class']
    lines = ['### Fault diagnosis (FDD)',
             f"**Predicted condition:** Class {cls} - {FDD_CLASS_NAMES.get(cls, '')}",
             f"**Confidence:** {r['confidence'] * 100:.1f}%"]
    if r['confidence'] < FDD_THRESHOLD:
        lines.append(f"**Low confidence** (below {FDD_THRESHOLD * 100:.1f}%): send for manual engineering review.")
    else:
        lines.append('High confidence.')

    low, high = max(0.0, r['rul_days'] - RUL_RADIUS), r['rul_days'] + RUL_RADIUS
    lines += ['', '### Remaining useful life (RUL)',
              f"**Estimated remaining life:** {r['rul_days']:.0f} days (about {r['rul_days'] / 30.4:.1f} months)",
              f'**Likely range:** {low:.0f} to {high:.0f} days (+/-{RUL_RADIUS:.0f} days; about 9 in 10 predictions fell inside this range in validation and testing)']
    if r['rul_days'] >= 0.9 * RUL_MAX_DAYS:
        lines.append(f'Note: the longest life in the training data is {RUL_MAX_DAYS:.0f} days, '
                     'so a prediction this high means "at least this long".')

    if true_fdd is not None or true_rul is not None:
        lines += ['', '### Actual values (from the dataset, for checking)']
        if true_fdd is not None:
            lines.append(f"**Actual condition:** Class {int(true_fdd)} - {FDD_CLASS_NAMES.get(int(true_fdd), '')}")
        if true_rul is not None:
            lines.append(f'**Actual remaining life:** {true_rul:.0f} days')
    return '\n\n'.join(lines)

def gas_chart(raw, title):
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    days = np.arange(raw.shape[0]) * STEP_TO_DAYS
    for ax, (g, gas) in zip(axes.ravel(), enumerate(app_pre.gas_columns)):
        ax.plot(days, raw[:, g], color='teal')
        ax.set_title(gas)
        ax.grid(True, linestyle='--', alpha=0.4)
    for ax in axes[1]:
        ax.set_xlabel('Days of monitoring')
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    return fig

def run_from_file(path, name, true_fdd=None, true_rul=None):
    try:
        raw = app_pre.validate(pd.read_csv(path, dtype=str), name=name)
    except txp.ValidationError as e:
        return f'### File rejected\n\nThe file could not be used: {e}', None
    except Exception as e:
        return f'### File rejected\n\nThe file could not be read as a CSV: {e}', None
    r = analyse(raw)
    return build_report(r, true_fdd, true_rul), gas_chart(raw, f'Gas readings: {name}')

def run_upload(file_path):
    if file_path is None:
        return 'Please upload a CSV file first.', None
    return run_from_file(file_path, os.path.basename(file_path))

def run_test_transformer(file_id):
    if not file_id:
        return 'Please choose a test transformer.', None
    return run_from_file(os.path.join(TEST_DIR, file_id), file_id,
                         TRUE_FDD.get(file_id), TRUE_RUL_DAYS.get(file_id))




# =====================================================================
# The interface
# =====================================================================
import gradio as gr

intro = (
    '# Transformer Health Check\n'
    'Analyses a power transformer from its dissolved gas readings (H2, CO, C2H4, C2H2) '
    'and reports its current condition and estimated remaining life.\n\n'
    'The input is the transformer\'s monitoring history: a CSV with **420 rows** (one reading every 12 hours, '
    'about 210 days) and the columns **H2, CO, C2H4, C2H2**.'
)

with gr.Blocks(title='Transformer Health Check', theme=gr.themes.Soft()) as demo:
    gr.Markdown(intro)
    with gr.Tab('Upload a CSV'):
        file_in = gr.File(label='Transformer readings (CSV)', file_types=['.csv'], type='filepath')
        upload_btn = gr.Button('Analyse', variant='primary')
    with gr.Tab('Try a test transformer'):
        sample_in = gr.Dropdown(choices=TEST_FILES, value=TEST_FILES[0] if TEST_FILES else None,
                                label='Test transformer (never seen during training)')
        sample_btn = gr.Button('Analyse', variant='primary')
    report_out = gr.Markdown()
    chart_out = gr.Plot(label='Gas readings over time')

    upload_btn.click(run_upload, inputs=file_in, outputs=[report_out, chart_out])
    sample_btn.click(run_test_transformer, inputs=sample_in, outputs=[report_out, chart_out])

    gr.Markdown('---\nDecision-support tool built for a student capstone project. '
                'It does not replace engineering judgement.')

if __name__ == '__main__':
    # PORT is set by the hosting platform (Azure App Service); 7860 is the local default
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)))
