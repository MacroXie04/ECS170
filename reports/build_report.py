"""Generate the four-page report from saved real experiment records.

Use the Codex bundled Python (reportlab installed), or install reportlab and run
python reports/build_report.py from any working directory. Edit TEAM below before submission.
"""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Image, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'result/stage_2_result'
DEST = ROOT / 'reports/Stage2_Report_Draft.pdf'
TEAM = '[Team name to be supplied]'
MEMBERS = [[f'Student {i}: [Name]', '[Student ID]', '[Email]'] for i in range(1, 5)]
INK = colors.HexColor('#182c45')
BLUE = colors.HexColor('#245b86')
BODY = ParagraphStyle('body', fontName='Times-Roman', fontSize=10, leading=11.5,
                      spaceAfter=6, textColor=colors.HexColor('#222222'))
HEAD = ParagraphStyle('head', parent=BODY, fontName='Times-Bold', textColor=INK,
                      spaceBefore=9, spaceAfter=5)
TITLE = ParagraphStyle('title', parent=HEAD, alignment=TA_CENTER, spaceBefore=0,
                       spaceAfter=8)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=10, leading=11)
story, text_source = [], []


def p(text, style=BODY):
    story.append(Paragraph(text, style))
    text_source.append(text)


def heading(text):
    p(text, HEAD)


def table(rows, widths):
    values = [[Paragraph(str(cell), SMALL) for cell in row] for row in rows]
    obj = Table(values, colWidths=widths, repeatRows=1, hAlign='LEFT')
    obj.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9eff5')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, BLUE),
        ('LINEBELOW', (0, -1), (-1, -1), 0.4, colors.HexColor('#b4c4d3')),
        ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(obj)
    story.append(Spacer(1, 6))
    text_source.extend([' | '.join(map(str, row)) for row in rows])


def picture(path, width):
    w, h = ImageReader(str(path)).getSize()
    story.append(Image(str(path), width=width, height=width*h/w))


def architecture():
    d = Drawing(504, 90)
    nodes = [(0, 'Input', '784 features'), (132, 'Hidden 1', '256 units'),
             (264, 'Hidden 2', '128 units'), (396, 'Output', '10 logits')]
    for x, title, label in nodes:
        d.add(Rect(x+2, 29, 104, 44, fillColor=colors.HexColor('#edf3f8'), strokeColor=BLUE))
        d.add(String(x+54, 56, title, textAnchor='middle', fontName='Times-Bold', fontSize=10))
        d.add(String(x+54, 40, label, textAnchor='middle', fontName='Times-Roman', fontSize=10))
        if x < 396:
            d.add(Line(x+108, 51, x+128, 51, strokeColor=BLUE))
            d.add(Polygon([x+128, 51, x+122, 54, x+122, 48], fillColor=BLUE))
    for x in [132, 264]:
        d.add(String(x+54, 13, 'ReLU + Dropout(0.2)', textAnchor='middle',
                     fontName='Times-Roman', fontSize=10))
    story.append(d)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman', 10)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(54, 30, 'ECS 170 | Stage 2 | Independent implementation: Zz_Wang | Draft')
    canvas.drawRightString(558, 30, str(doc.page))
    canvas.restoreState()


def main():
    records = {path.parent.name: json.loads(path.read_text())
               for path in RESULT.glob('*/record.json')}
    final, selected = records['final'], records['dropout']
    p('ECS 170 Artificial Intelligence - Fall 2026<br/>Course Project: Stage 2 Report', TITLE)
    p('<b>Draft:</b> experimental results are complete; team details and a published Stage 2 code link remain to be supplied.')
    heading('Team Information')
    p(TEAM)
    table([['Student name', 'Student ID', 'Email']] + MEMBERS, [200, 115, 189])
    heading('Section 1: Task Description')
    p('Implement a PyTorch multilayer perceptron (MLP) for the instructor-provided multiclass dataset, '
      'following the course module framework. The target was at least 95% test accuracy; this target '
      'was not assumed during development. The locked final model achieved 98.26% on the supplied test set.')
    heading('Section 2: Model Description')
    architecture()
    p('Figure 1. Selected architecture. Each hidden layer is fully connected and followed by ReLU '
      'and Dropout with probability 0.2. The output consists of raw logits. Cross-entropy combines '
      'log-softmax and negative log likelihood internally, so no output Softmax is applied during '
      'cross-entropy training. Prediction is the class with the largest logit.')
    heading('Section 3: Experiment Settings')
    heading('3.1 Dataset Description')
    p('The supplied train.csv contains 60,000 labeled instances and test.csv contains 10,000. '
      'There is no header. Each row has 785 integer entries: the first is a label in 0-9 and the '
      'remaining 784 are features in 0-255. Features are float32 divided by the fixed constant 255; '
      'labels are int64. No preprocessing parameters are estimated from validation or test data.')
    p('The instructor train/test boundary is preserved. A fixed stratified holdout from train.csv '
      'provides 54,000 fitting and 6,000 validation instances (split seed 42); all configurations '
      'use identical indices. No cross-validation is used. File-format auditing inspected both CSVs, '
      'but test predictions and metrics were excluded from selection.')
    heading('3.2 Detailed Experimental Setups')
    p('All validation experiments use batch size 128, up to 25 epochs, and seed 42 for Python, NumPy, '
      'PyTorch initialization and batch shuffling. Linear layers use PyTorch default initialization '
      '(uniform weights and biases bounded by the inverse square root of fan-in). Adam uses lr=0.001; '
      'the SGD comparison uses lr=0.05 and momentum=0.9. Weight decay and Dropout are zero unless specified. '
      'Each run creates a new model and optimizer; gradients are cleared each batch.')
    story.append(PageBreak())

    heading('3.2 Detailed Experimental Setups (continued)')
    p('At each epoch end, both fitting and validation metrics are measured in eval mode on the same '
      'weights. Validation-loss improvement must exceed 1e-6; patience is 5 epochs. The minimum-loss '
      'checkpoint is restored for each configuration. Configurations are ranked by validation accuracy '
      '(tie breakers: macro F1, then name). Loss magnitudes from different objectives are not compared.')
    p('The selected Dropout model had validation accuracy 97.95% at epoch 8. Its configuration and '
      'epoch count were locked before final training and test prediction. A fresh model was trained '
      'for 8 epochs on all 60,000 training instances. CPU execution used two threads, Python 3.9.6, '
      'PyTorch 2.8.0, NumPy 2.0.2, scikit-learn 1.6.1 and Matplotlib 3.9.4. Final fitting took '
      f"{final['training_seconds']:.2f} seconds, including per-epoch metric computation, but excluding "
      'initial CSV loading, post-training evaluation and artifact saving.')
    heading('3.3 Evaluation Metrics')
    p('Accuracy is the fraction of correctly classified instances. Per class, precision measures '
      'correct predictions among predicted positives, recall measures recovered true instances, '
      'and F1 is their harmonic mean. Macro averages weight all ten classes equally; weighted '
      'averages use true class support. Micro Precision/Recall/F1 equal accuracy for this complete '
      'single-label multiclass task. Undefined per-class divisions are set to zero. All labels '
      'are included explicitly in the metrics and confusion matrix.')
    heading('3.4 Source Code')
    p('Repository reference: <link href="https://github.com/MacroXie04/ECS170" color="#245b86">'
      'https://github.com/MacroXie04/ECS170</link>. This independent Stage 2 implementation is currently '
      'in the local Zz_Wang branch and has not been published to that repository. '
      '<b>Before submission, replace this status with a publicly accessible link to the actual '
      'Stage 2 source.</b> The repository main at inspection was 8144e75 and did not include this implementation.')
    p('Modules retain the template responsibilities: Dataset_Loader, Method_MLP, Setting, Evaluate '
      'and Result_Saver; Plot and the runner handle figures and configuration. The Stage 1 '
      'Softmax/cross-entropy pairing, train-mode override and reused-model KFold workflow are '
      'not carried into Stage 2. Saved records include seeds, split indices, configurations, '
      'data and source hashes, epoch histories, metrics, predictions and checkpoints.')
    heading('3.5 Training Convergence Plot')
    picture(RESULT / 'dropout/learning_curve.png', 504)
    p('Figure 2. Selected configuration on the internal fitting/validation split. Loss and accuracy '
      'are measured after each epoch. The dotted line marks epoch 8, whose weights were restored. '
      'Later epochs support the early-stopping decision and are not the final selected weights.')
    story.append(PageBreak())

    heading('3.5 Training Convergence Plot (continued)')
    picture(RESULT / 'final/learning_curve.png', 504)
    p('Figure 3. Fresh final-model training on all 60,000 examples for the previously locked eight '
      'epochs. There is no validation subset in this refit, and no per-epoch test curve was generated.')
    heading('3.6 Model Performance')
    rows = [['Model / data split', 'Accuracy', 'Macro P', 'Macro R', 'Macro F1']]
    for name, metrics in [('Selected / internal train', selected['train']),
                          ('Selected / validation', selected['validation']),
                          ('Final / full train', final['train']),
                          ('Final / test', final['test'])]:
        rows.append([name, f"{100*metrics['accuracy']:.4f}%"] +
                    [f"{metrics['macro'][k]:.6f}" for k in ['precision', 'recall', 'f1']])
    table(rows, [164, 85, 85, 85, 85])
    p('The final model correctly classified 9,826 of 10,000 test instances (174 errors). '
      'Test weighted Precision/Recall/F1 are 0.982634/0.982600/0.982594, respectively. '
      'Micro Precision/Recall/F1 are all 0.982600. Validation refers to the selected holdout model; '
      'the final refit has different weights and no separately measured validation performance.')
    picture(RESULT / 'final/confusion_matrix.png', 300)
    p('Figure 4. Final test confusion matrix. Rows are true labels; columns are predicted labels. '
      'Full per-class support and metrics are stored in final/record.json.')
    story.append(PageBreak())

    heading('3.7 Ablation Studies')
    p('Seven configurations were declared before comparison. Each used the fixed training-only '
      'holdout; no additional configuration was chosen after seeing the test result. Each row below '
      'describes a checkpoint selected by that configuration\'s own validation loss.')
    rows = [['Experiment / hidden units', 'Change from reference', 'Epoch', 'Val. Acc.', 'Macro F1']]
    descriptions = {
        'baseline': ('128', 'Adam + cross-entropy'),
        'depth2': ('256,128', 'Wider/deeper than baseline'),
        'depth3': ('256,128,64', 'Third hidden layer'),
        'squared_error': ('128', 'Probability squared error'),
        'sgd': ('128', 'SGD; lr 0.05; momentum 0.9'),
        'dropout': ('256,128', 'depth2 + Dropout 0.2'),
        'weight_decay': ('256,128', 'depth2 + decay 0.0001'),
    }
    for name, (units, change) in descriptions.items():
        r = records[name]
        rows.append([f'{name}: {units}', change, r['best_epoch'],
                     f"{100*r['validation']['accuracy']:.4f}%",
                     f"{r['validation']['macro']['f1']:.6f}"])
    table(rows, [135, 176, 43, 75, 75])
    p('Depth: the three-hidden-layer configuration did not improve validation accuracy over the '
      'baseline. Width and parameter count also change, so the comparison does not isolate a '
      'causal effect of depth. Regularization: Dropout improved over the matching unregularized '
      'two-hidden-layer model by 0.1833 percentage points; weight decay also improved that reference.')
    p('Loss: the squared-error objective computes the mean per-instance sum of squared differences '
      'between softmax probabilities and one-hot targets. It reached 97.5667% validation accuracy, '
      'below the cross-entropy baseline in this run. These loss values have different scales. '
      'Optimizer: the SGD setting slightly exceeded the Adam baseline, but its learning rate also '
      'differs; this is a comparison of specified settings rather than a universal optimizer ranking.')
    heading('Reliability and Limitations')
    p('The baseline improved from 9.95% initial validation accuracy to 97.80%, and targeted checks '
      'confirmed finite gradients, small-data fitting, preserved train/eval behavior, correct '
      'multiclass metrics and checkpoint consistency. An artifact audit confirmed identical '
      'non-overlapping holdout indices, recomputed saved metrics, and reproduced final test '
      'predictions from the saved checkpoint.')
    p('Only one seed and one holdout were used; small differences are not evidence of statistical '
      'significance. No claim of globally optimal hyperparameters is made. The final test score '
      'is a single locked-model evaluation, not a guarantee for future data. The final training '
      'accuracy of 99.4367% exceeds test accuracy by 1.1767 percentage points. The larger final '
      'training set and different held-out samples can explain why test accuracy exceeds the '
      'internal validation estimate; neither number substitutes for the other.')
    heading('References and Reproduction')
    p('Course sources: ECS_170_Fall_2026_Project_Description.docx.pdf (Stage 2), '
      'Stage_2-5_Report_Template.pdf, and the supplied data ReadMe. Implementation follows the '
      'course template; the original-template repository main was 465d116 at inspection.')
    p('PyTorch 2.8 CrossEntropyLoss: '
      '<link href="https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html" color="#245b86">official documentation</link>. '
      'scikit-learn precision_recall_fscore_support: '
      '<link href="https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.precision_recall_fscore_support.html" color="#245b86">official documentation</link>. '
      'Run script/stage_2_script/script_mlp.py with baseline, compare, final and plot in order, '
      'using a new --output directory. See STAGE2_GUIDE_ZH.md for setup and data flow.')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(DEST), pagesize=(612, 792), rightMargin=54, leftMargin=54,
                            topMargin=42, bottomMargin=48,
                            title='ECS 170 Stage 2 - Independent MLP Report Draft', author='Zz_Wang')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (DEST.parent / 'Stage2_Report_Draft.txt').write_text('\n\n'.join(text_source), encoding='utf-8')
    print(DEST)


if __name__ == '__main__':
    main()
