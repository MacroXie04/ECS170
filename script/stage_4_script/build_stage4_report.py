#!/usr/bin/env python3
"""Build a five-page report from verified experiment artifacts (no invented scores)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
INK = colors.HexColor('#17354A')
TEAL = colors.HexColor('#156A73')
PALE = colors.HexColor('#EAF3F4')


def read_json(path):
    return json.loads(path.read_text())


def audit_metrics(directory, metrics):
    with (directory / 'test_predictions.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    claimed = metrics['test']
    if metrics['task'] == 'classification':
        y, p = [int(r['label']) for r in rows], [int(r['prediction']) for r in rows]
        precision, recall, f1, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
        observed = dict(accuracy=accuracy_score(y, p), macro_precision=precision, macro_recall=recall,
                        macro_f1=f1, loss=np.mean([float(r['nll']) for r in rows]))
    else:
        count = sum(int(r['token_count']) for r in rows)
        loss = sum(float(r['nll_sum']) for r in rows) / count
        observed = dict(loss=loss, perplexity=math.exp(loss),
                        token_accuracy=sum(int(r['correct_tokens']) for r in rows) / count)
    for name, value in observed.items():
        if not math.isclose(value, claimed[name], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f'Metric evidence mismatch: {directory.name}/{name}')
    if len(rows) != claimed['n_examples']:
        raise ValueError('Test row count mismatch')
    return {'run': directory.name, 'rows': len(rows), 'metrics_recomputed': list(observed), 'passed': True}


def curves(runs, destination):
    palette = {'rnn': '#146C94', 'lstm': '#C36528', 'gru': '#34845A'}
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5))
    for axis, task in zip(axes, ('classification', 'generation')):
        for cell in ('rnn', 'lstm', 'gru'):
            run = runs[(task, cell)]
            h = run['history']
            axis.plot([r['epoch'] for r in h], [r['train_loss'] for r in h],
                      color=palette[cell], label=f'{cell.upper()} train')
            axis.plot([r['epoch'] for r in h], [r['validation_loss'] for r in h],
                      '--', color=palette[cell], label=f'{cell.upper()} validation')
        axis.set(xlabel='Epoch', ylabel='Cross-entropy loss', title=task.title())
        axis.grid(alpha=.2)
        axis.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(h_pad=2)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def architecture():
    drawing = Drawing(490, 115)
    labels = [('Token IDs', 5, 40, 75), ('Embedding\n128 dimensions', 98, 40, 94),
              ('RNN / LSTM / GRU\n128 hidden, 1 layer', 210, 40, 120)]
    for label, x, y, width in labels:
        drawing.add(Rect(x, y, width, 47, fillColor=PALE, strokeColor=TEAL, strokeWidth=.7, rx=5, ry=5))
        parts = label.split('\n')
        for i, line in enumerate(parts):
            drawing.add(String(x + width / 2, y + 27 - i * 12, line, textAnchor='middle', fontName='Helvetica', fontSize=10))
    for x1, x2 in ((80, 98), (192, 210)):
        drawing.add(Line(x1, 64, x2, 64, strokeColor=TEAL))
        drawing.add(Polygon([x2, 64, x2-4, 67, x2-4, 61], fillColor=TEAL, strokeColor=TEAL))
    for y, text in ((83, 'Masked mean -> 2 logits'), (23, 'Each state -> V logits')):
        drawing.add(Line(330, 64, 345, y, strokeColor=TEAL))
        drawing.add(Rect(345, y-15, 143, 32, fillColor=PALE, strokeColor=TEAL, strokeWidth=.7, rx=5, ry=5))
        drawing.add(String(416.5, y-2, text, textAnchor='middle', fontName='Helvetica', fontSize=10))
    drawing.add(String(417, 108, 'Sentiment classification', textAnchor='middle', fontSize=10, fillColor=INK))
    drawing.add(String(417, 0, 'Next-token generation', textAnchor='middle', fontSize=10, fillColor=INK))
    return drawing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'result/stage_4_result/runs')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/pdf/Stage_4_Report_Draft.pdf')
    parser.add_argument('--team-info', type=Path, default=ROOT / 'data/stage_4_data/team_information.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    team = read_json(args.team_info)
    artifacts = ROOT / 'result/stage_4_result/report'
    artifacts.mkdir(parents=True, exist_ok=True)
    runs, audit = {}, []
    for task in ('classification', 'generation'):
        for cell in ('rnn', 'lstm', 'gru'):
            directory = args.runs / f'{task}_{cell}_baseline'
            metrics, config = read_json(directory / 'metrics.json'), read_json(directory / 'config.json')
            if metrics['smoke']:
                raise ValueError('Smoke results cannot enter the report')
            audit.append(audit_metrics(directory, metrics))
            runs[task, cell] = dict(metrics=metrics, config=config, history=read_json(directory / 'history.json'), directory=directory)
    (artifacts / 'metric_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    summary = [run['metrics'] for run in runs.values()]
    (artifacts / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    curve_path = artifacts / 'convergence.png'
    curves(runs, curve_path)
    classification = runs['classification', 'rnn']['config']['data_metadata']
    generation = runs['generation', 'rnn']['config']['data_metadata']
    base = ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=12, textColor=INK, spaceAfter=6)
    heading = ParagraphStyle('heading', parent=base, fontName='Helvetica-Bold', fontSize=12, leading=15, spaceBefore=10, spaceAfter=6)
    title = ParagraphStyle('title', parent=base, fontName='Helvetica-Bold', fontSize=19, leading=23, spaceAfter=10)
    small = ParagraphStyle('small', parent=base, fontSize=10, leading=12)
    story = []
    def p(text, style=base):
        return Paragraph(text, style)
    def para(text):
        story.append(p(text))
    def h(text):
        story.append(p(text, heading))
    def table(rows, widths):
        cells = [[p(escape(str(cell)), small) for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        item.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), PALE), ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.8,TEAL), ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D8E1E5')),
            ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ]))
        story.append(item)
        story.append(Spacer(1, 7))
    width = 504
    story.append(p('ECS 170 | Fall 2026', heading))
    story.append(p('Course Project: Stage 4', title))
    para('<b>Harmony Pangu Large Models</b><br/>Text classification and generation with recurrent neural networks')
    table([['Team member', 'Student ID', 'Email']] + team['members'], [135, 110, 259])
    para('<b>Working report, prepared November 12, 2026.</b> The experiments below cover Hongzhe\'s shared Stage 4 implementation. They do not assert that other team members completed separate implementations. Source publication and team review remain pending.')
    h('Section 1: Task Description')
    para('We classify IMDb movie reviews as negative or positive and learn next-token models for short jokes. For each task, we train a vanilla tanh RNN, an LSTM and a GRU from scratch. Sentiment is evaluated on the instructor\'s official test split. Text generation continues a prompt consisting of the first three words of a training joke and is compared with the source text.')
    h('Section 2: Model Description')
    story.append(architecture())
    story.append(Spacer(1, 6))
    para('Figure 1. Both tasks share a learned embedding and a causal, unidirectional recurrent layer. Classification averages only valid token states, applies dropout and predicts two raw logits. Generation maps each recurrent state to vocabulary logits; training predicts the next token, including EOS, while excluding PAD targets. No sequence crosses a joke boundary.')
    para('The cell type is the controlled architecture ablation. LSTM introduces cell state and gates; GRU uses gated hidden state; the vanilla RNN uses a tanh recurrence. Dimensions, input data, optimizer and selection policy remain fixed within each task. Parameter counts differ because gated cells contain more matrices.')

    story.append(PageBreak())
    h('Section 3: Experiment Settings')
    h('3.1 Dataset Description')
    table([['Dataset', 'Train', 'Validation', 'Test', 'Vocabulary'],
        ['IMDb', *[classification['counts'][s] for s in ('train','validation','test')], classification['vocabulary_size']],
        ['Short jokes', *[generation['counts'][s] for s in ('train','validation','test')], generation['vocabulary_size']]],
        [130, 86, 96, 86, 106])
    overlaps = classification['normalized_text_overlap_counts']
    para(f'IMDb contains 25,000 official training and 25,000 official test reviews, balanced by label. We form a validation set from official training only, grouping identical normalized reviews together. Training and validation have {overlaps["train_validation"]} shared normalized texts. The supplied data itself contains {overlaps["train_test"]} distinct normalized texts shared between our training and official test, and {overlaps["validation_test"]} between validation and test; we preserve the required official test and disclose this limitation.')
    para('The joke CSV has 1,622 rows and 1,609 normalized-text groups. Thirteen duplicate rows are retained, with each group assigned wholly to one of the approximately 80%/10%/10% train/validation/test splits. No normalized joke spans splits. Vocabulary is fitted on training text only for both tasks. Cleaning decodes HTML, normalizes Unicode, lowercases text, and preserves punctuation, negation and contractions.')
    trunc = classification['truncated_counts']
    gen_targets = runs['generation','rnn']['metrics']['test']['n_tokens']
    gen_unknown = generation['unknown_token_counts']['test']
    para(f'IMDb sequences retain their first 256 tokens; {trunc["train"]:,} training, {trunc["validation"]:,} validation and {trunc["test"]:,} test reviews are truncated. Jokes remain complete with BOS/EOS boundaries. Unseen words map to UNK. The joke test set has {gen_unknown} UNK targets out of {gen_targets:,} predicted targets ({100*gen_unknown/gen_targets:.2f}%, excluding BOS and including EOS); this limits interpretation of language-model scores.')
    h('3.2 Detailed Experimental Setups')
    table([['Setting', 'Both tasks / task-specific values'],
        ['Architecture', 'Embedding 128; hidden 128; 1 recurrent layer; dropout 0.30 on embeddings and output representations'],
        ['Optimization', 'Adam, learning rate 0.001, batch 128, weight decay 0; global gradient norm clipped to 1.0'],
        ['Training budget', 'IMDb: at most 12 epochs, patience 4. Jokes: at most 80 epochs, patience 12. Lowest validation loss selects checkpoint; earliest exact tie.'],
        ['Randomness / precision', 'Seed 1704 for splits, initialization and loader order; float32 parameters with cuDNN TF32 allowed; deterministic cuDNN requested. One seed per model.'],
        ['Initialization', 'PyTorch defaults: embedding N(0,1), zero PAD row; recurrent weights/biases uniform +/-1/sqrt(128); Linear default Kaiming-uniform.'],
        ['Hardware / software', 'NVIDIA RTX PRO 4500 Blackwell Server Edition (32 GB); PyTorch 2.14.1+cu130; Python 3.12.']], [112, 392])
    h('3.3 Evaluation Metrics')
    para('Classification reports accuracy and macro-averaged precision, recall and F1 across both classes. Generation reports token-weighted cross-entropy, perplexity = exp(cross-entropy), and next-token accuracy under teacher forcing, including EOS/UNK targets and excluding PAD/BOS targets. These quantify prediction of reference continuations, not humor, factual correctness or free-running coherence.')

    story.append(PageBreak())
    h('3.4 Source Code and Reproducibility')
    para('The implementation follows the instructor layout: Dataset_Loader extends the dataset base and SequenceModel extends the method base and torch.nn.Module. The experiment runner handles fitting, validation selection and evidence export without overriding nn.Module.train(). Source lives on code-server-gpu in /home/ubuntu/PycharmProjects/ECS170, branch stage4-rnn-hongzhe. No pull request has been opened. An accessible submission code link must be added after source publication is approved; this is a review draft.')
    para('Run: <font name="Courier">.venv-cuda/bin/python script/stage_4_script/run_stage4.py --device cuda</font>. Each run stores source SHA256 values, configuration, vocabulary/split provenance, environment, epoch history, best/last checkpoints and raw held-out predictions. Repeating the same command resumes an interrupted run; mismatched configurations or source hashes are rejected. CPU unit tests cover data separation, padding/causality, generation and checkpoint behavior.')
    h('3.5 Training Convergence Plot')
    story.append(Image(str(curve_path), width=476, height=442))
    para('Figure 2. Solid lines show training loss with dropout active; dashed lines show validation loss in evaluation mode. Early stopping and the maximum epoch budget can produce different curve lengths. Checkpoints are selected independently for each model using validation loss only; test scores do not select an epoch.')

    story.append(PageBreak())
    h('3.6 Model Performance')
    para('<b>IMDb sentiment classification.</b> All scores below use the complete, unchanged 25,000-review official test set. Precision, recall and F1 are macro averages; percentages are rounded only for display.')
    rows = [['Cell', 'Best epoch', 'Accuracy', 'Precision', 'Recall', 'F1']]
    for cell in ('rnn','lstm','gru'):
        m = runs['classification',cell]['metrics']; t=m['test']
        rows.append([cell.upper(), m['best_epoch'], *[f'{100*t[k]:.2f}%' for k in ('accuracy','macro_precision','macro_recall','macro_f1')]])
    table(rows, [69, 79, 89, 89, 89, 89])
    para('<b>Joke next-token prediction.</b> Every model uses the same held-out joke rows and training vocabulary. Perplexity is over the resulting vocabulary with OOV words mapped to UNK; values are not directly comparable to language models using other tokenizers or vocabularies.')
    rows = [['Cell', 'Best epoch', 'Test loss', 'Perplexity', 'Token accuracy']]
    for cell in ('rnn','lstm','gru'):
        m=runs['generation',cell]['metrics'];t=m['test']
        rows.append([cell.upper(),m['best_epoch'],f'{t["loss"]:.4f}',f'{t["perplexity"]:.2f}',f'{100*t["token_accuracy"]:.2f}%'])
    table(rows, [83, 84, 112, 113, 112])
    h('3.7 Ablation Studies: Recurrent Cell')
    best_cls = max(('rnn','lstm','gru'),key=lambda c:runs['classification',c]['metrics']['test']['accuracy'])
    best_gen = min(('rnn','lstm','gru'),key=lambda c:runs['generation',c]['metrics']['test']['loss'])
    para(f'{best_cls.upper()} has the highest observed classification test accuracy and {best_gen.upper()} the lowest observed generation test loss in this six-run comparison. These are descriptive results after validation-based epoch selection. A single seed does not establish statistical significance or a universal ranking; differences in parameter count are intrinsic to the cell change.')
    rows = [['Task / cell', 'Parameters', 'Epochs run', 'Train + val seconds']]
    for task in ('classification','generation'):
        for cell in ('rnn','lstm','gru'):
            m=runs[task,cell]['metrics']
            rows.append([f'{task.title()} / {cell.upper()}',f'{m["parameters"]:,}',m['epochs_completed'],f'{m["train_seconds"]:.1f}'])
    table(rows,[175,108,93,128])
    para('Time includes training and validation passes; it excludes data preparation, checkpoint writing, final test evaluation and plotting. Runs are sequential on the same GPU. First-use backend overhead may affect the first run, so these timings should not be interpreted as a controlled hardware benchmark. Every displayed numeric test metric was recomputed from the saved prediction CSV before report generation.')

    story.append(PageBreak())
    h('3.8 Three-word Prompt Generation and Error Analysis')
    examples={c:read_json(runs['generation',c]['directory']/'generation_examples.json') for c in ('rnn','lstm','gru')}
    para('The three prompts below are the first three words of actual training jokes. For each prefix, the shortest matching training reference (with ID tie-break) is selected independently of model output. Greedy decoding allows at most 40 new tokens, stops at EOS and blocks PAD/BOS/UNK outputs. All generated text is shown as produced, without manual correction.')
    for i, reference in enumerate(examples['rnn']):
        h(f'Prompt {i+1}: {escape(reference["prefix"])}')
        para(f'<b>Training reference ({escape(reference["reference_id"])}):</b> {escape(reference["reference"])}')
        rows=[['Cell','Greedy continuation (including prompt)']]
        for cell in ('rnn','lstm','gru'):
            rows.append([cell.upper(), examples[cell][i]['greedy']])
        table(rows,[65,439])
    para('<b>Interpretation.</b> The RNN repeats punctuation in the first example and combines incompatible question templates. LSTM and GRU produce more recognizable sentence frames, but their punchlines remain incomplete or logically unrelated. None reconstructs the displayed reference exactly. Because each three-word prefix can start many jokes, reference mismatch alone is not an error; coherence and a meaningful punchline also matter. These are seen-prefix reconstruction examples, not independent test-set generalization.')
    para('The saved generation_examples.json also contains seeded top-k samples (temperature 0.8, k=20, maximum 40 new tokens). They demonstrate decoding variability but are not used for model selection. The small corpus, many rare words, substantial held-out OOV rate and a single random seed constrain the conclusions. Larger or pretrained models, multi-seed repeats and human humor ratings were outside this experiment.')
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL); canvas.setLineWidth(.6)
        canvas.line(54, 40, 558, 40)
        canvas.setFont('Helvetica', 9); canvas.setFillColor(INK)
        canvas.drawString(54, 27, 'ECS 170 | Stage 4 | Working report')
        canvas.drawRightString(558, 27, str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(args.output),pagesize=letter,rightMargin=54,leftMargin=54,topMargin=40,bottomMargin=50,
                          title='ECS 170 Stage 4 - Harmony Pangu Large Models',author='Hongzhe Xie')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    pages=len(PdfReader(args.output).pages)
    if pages != 5:
        raise RuntimeError(f'Report must be exactly five pages, got {pages}; adjust layout before delivery')
    print(json.dumps({'report':str(args.output),'pages':pages,'verified_runs':len(audit)}))


if __name__=='__main__':
    main()
