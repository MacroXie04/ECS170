import { window } from '@kit.ArkUI';
import { NativeModel, NativeTensor } from './NativeModel';
import batchprep from 'libbatchprep.so';
import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { common } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { fileIo } from '@kit.CoreFileKit';
class Config {
 id: string = ''; modelId: string = ''; dataset: string = ''; height: number = 28; width: number = 28;
 inputChannels: number = 1; classes: number = 10; batchSize: number = 128; epochs: number = 15;
 learningRate: number = 0.001; optimizer: string = 'adam'; trainCount: number = 0; valCount: number = 0; testCount: number = 0;
 mean: number[] = []; std: number[] = []; replicate: boolean = false; action: string = 'smoke'; smokeSteps: number = 2;
 backend: string = 'cpu'; probeTrain: boolean = true; probeModel: string = ''; probeSteps: number = 3;
}
class BackendProbe {
 id: string = ''; requestedBackend: string = ''; training: boolean = false; state: string = 'starting';
 built: boolean = false; steps: number = 0; stepMs: number[] = []; changedWeights: number = 0;
 elapsedMs: number = 0; capabilities: string = ''; librariesBefore: string = ''; librariesAfter: string = ''; error: string = '';
}
class Variables { variables: string[] = []; learned: string[] = []; }
class StateTensor { name: string = ''; shape: number[] = []; offset: number = 0; bytes: number = 0; }
class Evaluation { loss: number = 0; accuracy: number = 0; logits: Float32Array = new Float32Array(0); }
class Epoch { epoch: number = 0; trainEvalLoss: number = 0; trainEvalAccuracy: number = 0; valLoss: number = 0; valAccuracy: number = 0; learningRate: number = 0; optimizerMs: number = 0; epochMs: number = 0; elapsedMs: number = 0; }
class Report {
 id: string = ''; state: string = 'running'; formal: boolean = false; steps: number = 0; seen: number = 0;
 optimizerMs: number = 0; trainingMs: number = 0; evaluationMs: number = 0; elapsedMs: number = 0;
 initialTrainLoss: number = 0; bestEpoch: number = 0; bestValAccuracy: number = -1; bestValLoss: number = 1e30;
 changedWeights: number = 0; testLoss: number = 0; testAccuracy: number = 0; epochs: Epoch[] = []; error: string = '';
}
@Entry
@Component
struct Index {
 @State message: string = 'Stage 3 · 鸿蒙本地训练';
 @State detail: string = '等待训练任务';
 @State timer: string = '00:00:00';
 private dir: string = ''; private c: Config = new Config(); private vars: Variables = new Variables();
 private displayedElapsed: number = -1; private clockStart: number = 0; private timerId: number = -1; private started: boolean = false;
 private x: Float32Array = new Float32Array(0); private y: Float32Array = new Float32Array(0);
 private empty: ArrayBuffer = new ArrayBuffer(0); private batch: number = 0; private currentLR: number = 0;
 aboutToAppear(): void { if(!this.started){this.started=true;this.run();} }
 aboutToDisappear(): void { if(this.timerId>=0)clearInterval(this.timerId); }
 updateTimer(): void {const s=Math.floor((this.displayedElapsed>=0?this.displayedElapsed:batchprep.now()-this.clockStart)/1000);this.timer=Math.floor(s/3600).toString().padStart(2,'0')+':'+Math.floor(s/60%60).toString().padStart(2,'0')+':'+(s%60).toString().padStart(2,'0');}
 log(s: string): void {hilog.info(0x170,'Stage3Train','%{public}s',this.c.id+' '+s);}
 save(n: string,data: ArrayBuffer|string): void {const p=this.dir+'/'+n;const f=fileIo.openSync(p+'.tmp',fileIo.OpenMode.CREATE|fileIo.OpenMode.READ_WRITE|fileIo.OpenMode.TRUNC);fileIo.writeSync(f.fd,data);fileIo.closeSync(f);fileIo.renameSync(p+'.tmp',p);}
 read(n: string): ArrayBuffer {const p=this.dir+'/'+n;const b=new ArrayBuffer(fileIo.statSync(p).size);const f=fileIo.openSync(p,fileIo.OpenMode.READ_ONLY);const count=fileIo.readSync(f.fd,b);fileIo.closeSync(f);if(count!==b.byteLength)throw new Error('Incomplete file '+n);return b;}
 async yieldUI(): Promise<void> {this.updateTimer();await new Promise<void>((resolve)=>setTimeout(resolve,1));}
 configure(n: number): void {if(this.batch===n)return;const c=this.c;this.batch=n;this.x=new Float32Array(n*c.inputChannels*c.height*c.width);this.y=new Float32Array(n*c.classes);batchprep.configure(c.height,c.width,c.inputChannels,c.classes,n,new Float64Array(c.mean).buffer as ArrayBuffer,new Float64Array(c.std).buffer as ArrayBuffer);}
 fill(model: NativeModel,images: ArrayBuffer,labels: ArrayBuffer,schedule: ArrayBuffer,start: number,training: boolean,n: number): NativeTensor[] {
  this.configure(n);batchprep.fillScheduled(images,labels,schedule,this.x.buffer as ArrayBuffer,this.y.buffer as ArrayBuffer,start,this.c.replicate,training);
  const inputs=model.getInputs();if(!inputs||inputs.length===0)throw new Error('Model has no inputs');
  for(let i=0;i<inputs.length;i++){if(inputs[i].elementNum===this.x.length){if(inputs[i].shape.length!==4||inputs[i].shape[1]!==this.c.inputChannels||inputs[i].shape[2]!==this.c.height||inputs[i].shape[3]!==this.c.width)throw new Error('Expected NCHW input '+inputs[i].shape);inputs[i].setData(this.x.buffer as ArrayBuffer);}else if(inputs[i].elementNum===this.y.length)inputs[i].setData(this.y.buffer as ArrayBuffer);else if(inputs[i].elementNum===1)inputs[i].setData(new Float32Array([this.currentLR]).buffer as ArrayBuffer);else throw new Error('Input mismatch '+inputs[i].name+' '+inputs[i].shape);}
  return inputs;
 }
 async load(mode: string): Promise<NativeModel> {
  const m=await NativeModel.load(this.read(this.c.modelId+'_'+mode+'.ms'),true);
  if(!m.getInputs()||m.getInputs().length===0)throw new Error('Cannot load '+mode+' graph');
  if(mode!=='eval')m.trainMode=true;
  return m;
 }
 copy(source: NativeModel,target: NativeModel): void {
  const sw=source.getWeights();const tw=target.getWeights();const names: string[]=[];for(let i=0;i<tw.length;i++)names.push(tw[i].name);
  const selected: NativeTensor[]=[];
  for(let i=0;i<sw.length;i++)if(this.vars.variables.indexOf(sw[i].name)>=0&&names.indexOf(sw[i].name)>=0)selected.push(sw[i]);
  if(selected.length===0||!target.updateWeights(selected))throw new Error('State copy failed');
 }
 setLR(m: NativeModel,lr: number): void {this.currentLR=lr;m.learningRate=lr;if(Math.abs(m.learningRate-lr)>Math.max(1e-9,lr*1e-5))throw new Error('Learning rate setter failed: '+m.learningRate+' expected '+lr);}
 snapshot(m: NativeModel,suffix: string): Float32Array {
  const ws=m.getWeights();const kept: NativeTensor[]=[];let count=0;
  for(let i=0;i<ws.length;i++)if(this.vars.learned.indexOf(ws[i].name)>=0){kept.push(ws[i]);count+=ws[i].elementNum;}
  if(count===0)throw new Error('No learned weights');const out=new Float32Array(count);let offset=0;
  for(let i=0;i<kept.length;i++){const values=new Float32Array(kept[i].getData());out.set(values,offset);offset+=values.length;}
  this.save(this.c.id+'_'+suffix+'.bin',out.buffer as ArrayBuffer);return out;
 }
 saveState(m: NativeModel): void {
  const ws=m.getWeights();const meta: StateTensor[]=[];const arrays: Uint8Array[]=[];let size=0;
  for(let i=0;i<ws.length;i++)if(this.vars.variables.indexOf(ws[i].name)>=0){const b=new Uint8Array(ws[i].getData());const item=new StateTensor();item.name=ws[i].name;item.shape=ws[i].shape;item.offset=size;item.bytes=b.byteLength;meta.push(item);arrays.push(b);size+=b.byteLength;}
  const all=new Uint8Array(size);for(let i=0;i<arrays.length;i++)all.set(arrays[i],meta[i].offset);
  this.save(this.c.id+'_best_state.bin',all.buffer as ArrayBuffer);this.save(this.c.id+'_best_state.json',JSON.stringify(meta));
 }
 async evaluator(train: NativeModel,clone: NativeModel): Promise<NativeModel> {
  clone.destroy();const fresh=await this.load('eval');this.copy(train,fresh);fresh.trainMode=false;return fresh;
 }
 async evaluate(m: NativeModel,images: ArrayBuffer,labelsBuffer: ArrayBuffer): Promise<Evaluation> {
  const labels=new Uint8Array(labelsBuffer);const c=this.c;const result=new Evaluation();result.logits=new Float32Array(labels.length*c.classes);let correct=0;
  for(let start=0;start<labels.length;start+=c.batchSize){
   const inputs=this.fill(m,images,labelsBuffer,this.empty,start,false,c.batchSize);const outputs=await m.predict(inputs);let values=new Float32Array(0);
   for(let i=0;i<outputs.length;i++)if(outputs[i].elementNum===c.batchSize*c.classes){values=new Float32Array(outputs[i].getData());break;}
   if(values.length===0){for(let i=0;i<outputs.length;i++)this.log('EVAL_OUTPUT '+outputs[i].name+' '+outputs[i].shape);throw new Error('No logits in evaluation graph');}
   const actual=Math.min(c.batchSize,labels.length-start);result.logits.set(values.subarray(0,actual*c.classes),start*c.classes);
   for(let b=0;b<actual;b++){let peak=-Infinity;let arg=0;for(let j=0;j<c.classes;j++){const v=values[b*c.classes+j];if(!Number.isFinite(v))throw new Error('Nonfinite logits');if(v>peak){peak=v;arg=j;}}
    let sum=0;for(let j=0;j<c.classes;j++)sum+=Math.exp(values[b*c.classes+j]-peak);result.loss+=Math.log(sum)+peak-values[b*c.classes+labels[start+b]];if(arg===labels[start+b])correct++;}
   if(start%(c.batchSize*10)===0)await this.yieldUI();
  }
  result.loss/=labels.length;result.accuracy=correct/labels.length;return result;
 }
 async reload(): Promise<void> {
  const c=this.c;const model=await NativeModel.load(this.read(c.id+'_best.ms'),false);
  const split=c.action==='reload_smoke'?'val':'test';const r=await this.evaluate(model,this.read(c.dataset+'_'+split+'_images.bin'),this.read(c.dataset+'_'+split+'_labels.bin'));const expected=new Float32Array(this.read(c.id+'_'+split+'_logits.bin'));
  if(expected.length!==r.logits.length)throw new Error('Reload output length mismatch');let maxError=0;for(let i=0;i<expected.length;i++)maxError=Math.max(maxError,Math.abs(expected[i]-r.logits[i]));
  this.save(c.id+'_reloaded_logits.bin',r.logits.buffer as ArrayBuffer);this.save(c.id+'_reload.json',JSON.stringify({passed:maxError<0.0001,maxError:maxError,scope:split,loss:r.loss,accuracy:r.accuracy}));
  const saved: Report=JSON.parse(fileIo.readTextSync(this.dir+'/'+c.id+'_result.json')) as Report;this.displayedElapsed=saved.elapsedMs;
  this.message=c.dataset.toUpperCase()+' · '+(saved.formal?'训练':'预检')+'与重载验证'+(maxError<0.0001?'通过':'失败');
  this.detail='完成 '+saved.epochs.length+' 轮 · '+(saved.formal?'测试':'验证')+'准确率 '+(r.accuracy*100).toFixed(2)+'%\n训练步骤耗时 '+(saved.optimizerMs/60000).toFixed(2)+' 分钟\n上方计时为任务总耗时（含评估与保存）';this.log('RELOAD '+maxError);
 }
 async train(report: Report): Promise<void> {
  const c=this.c;const train=await this.load('train');let clone=await this.load('eval');const saved=await this.load('eval');saved.trainMode=false;let tail=train;
  const remainder=c.trainCount%c.batchSize;if(remainder>0)tail=await this.load('tail');
  const images=this.read(c.dataset+'_train_images.bin');const labels=this.read(c.dataset+'_train_labels.bin');const vx=this.read(c.dataset+'_val_images.bin');const vy=this.read(c.dataset+'_val_labels.bin');
  if(labels.byteLength!==c.trainCount||vy.byteLength!==c.valCount)throw new Error('Data count mismatch');
  const before=this.snapshot(train,'weights_before');clone=await this.evaluator(train,clone);let ev=clone;
  const initial=await this.evaluate(ev,images,labels);report.initialTrainLoss=initial.loss;this.log('INITIAL_TRAIN_CE '+initial.loss);
  const epochs=c.action==='smoke'?3:c.epochs;report.formal=c.action==='train';
  for(let epoch=1;epoch<=epochs;epoch++){
   const epochStart=batchprep.now();const stepBefore=report.optimizerMs;const lr=c.learningRate*(1+Math.cos(Math.PI*(epoch-1)/c.epochs))/2;train.trainMode=true;this.setLR(train,lr);this.log('TRAIN_STATE mode='+train.trainMode+' lr='+train.learningRate);
   const epochStepStart=report.steps;const schedule=this.read(c.modelId+'_epoch_'+epoch+'.bin');const trainStart=batchprep.now();
   for(let start=0;start<c.trainCount;start+=c.batchSize){
    const size=Math.min(c.batchSize,c.trainCount-start);const active=size===c.batchSize?train:tail;
    if(active!==train){this.copy(train,active);active.trainMode=true;this.setLR(active,lr);}
    const inputs=this.fill(active,images,labels,schedule,start,true,size);
    if(epoch===1&&start===0){const expected=new Float32Array(this.read(c.modelId+'_expected_batch.bin'));if(expected.length!==this.x.length)throw new Error('Reference batch size mismatch');let error=0;for(let i=0;i<expected.length;i++)error=Math.max(error,Math.abs(expected[i]-this.x[i]));if(error>1e-6)throw new Error('Preprocess mismatch '+error);this.log('PREPROCESS_MATCH '+error);}
    const stepStart=batchprep.now();if(!active.runStep(inputs))throw new Error('Training step failed');report.optimizerMs+=batchprep.now()-stepStart;report.steps++;report.seen+=size;
    if(active!==train)this.copy(active,train);
    this.message=c.dataset.toUpperCase()+' · 第 '+epoch+'/'+c.epochs+' 轮';this.detail='本轮训练 '+Math.min(start+size,c.trainCount)+' / '+c.trainCount+' 张图像\n累计训练步骤耗时 '+(report.optimizerMs/1000).toFixed(1)+' 秒';await this.yieldUI();
    if(c.action==='smoke'&&report.steps-epochStepStart>=c.smokeSteps){if(remainder>0&&size===c.batchSize){start=c.trainCount-remainder-c.batchSize;}else break;}
   }
   report.trainingMs+=batchprep.now()-trainStart;const evalStart=batchprep.now();clone=await this.evaluator(train,clone);ev=clone;
   const tr=await this.evaluate(ev,images,labels);const validation=await this.evaluate(ev,vx,vy);report.evaluationMs+=batchprep.now()-evalStart;
   const row=new Epoch();row.epoch=epoch;row.trainEvalLoss=tr.loss;row.trainEvalAccuracy=tr.accuracy;row.valLoss=validation.loss;row.valAccuracy=validation.accuracy;row.learningRate=lr;row.optimizerMs=report.optimizerMs-stepBefore;row.epochMs=batchprep.now()-epochStart;row.elapsedMs=batchprep.now()-this.clockStart;report.epochs.push(row);
   if(validation.accuracy>report.bestValAccuracy||(validation.accuracy===report.bestValAccuracy&&validation.loss<report.bestValLoss)){
    report.bestEpoch=epoch;report.bestValAccuracy=validation.accuracy;report.bestValLoss=validation.loss;
    this.copy(train,saved);this.saveState(train);if(c.action==='smoke')this.save(c.id+'_val_logits.bin',validation.logits.buffer as ArrayBuffer);
   }
   report.elapsedMs=batchprep.now()-this.clockStart;this.save(c.id+'_progress.json',JSON.stringify(report));this.log('EPOCH '+JSON.stringify(row));
  }
  const after=this.snapshot(train,'weights_after');if(before.length!==after.length)throw new Error('Weight count changed');for(let i=0;i<before.length;i++)if(before[i]!==after[i])report.changedWeights++;
  if(report.changedWeights===0)throw new Error('No weight updates');
  saved.trainMode=false;const selected=await this.evaluate(saved,vx,vy);if(selected.accuracy!==report.bestValAccuracy||Math.abs(selected.loss-report.bestValLoss)>1e-4)throw new Error('Saved best validation mismatch');
  if(!saved.exportModel(this.dir+'/'+c.id+'_best.ms',mindSporeLite.QuantizationType.NO_QUANT,true))throw new Error('Best inference export failed');
  if(c.action==='train'){
   const best=await NativeModel.load(this.read(c.id+'_best.ms'),false);
   const t=await this.evaluate(best,this.read(c.dataset+'_test_images.bin'),this.read(c.dataset+'_test_labels.bin'));report.testLoss=t.loss;report.testAccuracy=t.accuracy;this.save(c.id+'_test_logits.bin',t.logits.buffer as ArrayBuffer);
  }
  report.state=c.action==='train'?'completed':'smoke_passed';report.elapsedMs=batchprep.now()-this.clockStart;this.save(c.id+'_result.json',JSON.stringify(report));this.message=c.dataset.toUpperCase()+' · '+(c.action==='train'?'训练完成':'预检完成');this.log('COMPLETE '+JSON.stringify(report));
 }
 async diagnoseOptimizer(): Promise<void> {
  const model=await NativeModel.load(this.read(this.c.optimizer==='sgd'?'optimizer_sgd_diagnostic.ms':'optimizer_diagnostic.ms'),true);model.trainMode=true;
  const rows: number[][]=[];
  for(let step=0;step<=3;step++){
   if(step>0){model.learningRate=this.c.learningRate/step;const inputs=model.getInputs();for(let i=0;i<inputs.length;i++){this.log('DIAG_INPUT '+inputs[i].name+' '+inputs[i].shape);if(inputs[i].elementNum===3)inputs[i].setData(new Float32Array([0.2,-0.3,0.4]).buffer as ArrayBuffer);else if(inputs[i].shape.length===2)inputs[i].setData(new Float32Array([0.5]).buffer as ArrayBuffer);else inputs[i].setData(new Float32Array([this.c.learningRate/step]).buffer as ArrayBuffer);}if(!model.runStep(inputs))throw new Error('Diagnostic step failed');}
   const ws=model.getWeights();for(let i=0;i<ws.length;i++)if(ws[i].name==='weight'){const values=new Float32Array(ws[i].getData());const row:number[]=[];for(let j=0;j<values.length;j++)row.push(values[j]);rows.push(row);this.log('DIAG_WEIGHT '+JSON.stringify(row));}
  }
  this.save('optimizer_diagnostic.json',JSON.stringify(rows));this.save(this.c.id+'_diagnostic.json',JSON.stringify({completed:true}));this.message='Optimizer diagnostic completed';
 }
 async diagnoseBN(): Promise<void> {
  const model=await NativeModel.load(this.read('bn_diagnostic.ms'),true);model.trainMode=true;
  model.learningRate=0.01;const inputs=model.getInputs();for(let i=0;i<inputs.length;i++){this.log('BN_INPUT '+inputs[i].name+' '+inputs[i].shape);if(inputs[i].elementNum===1)inputs[i].setData(new Float32Array([0.01]).buffer as ArrayBuffer);else if(inputs[i].name==='x')inputs[i].setData(this.read('bn_x.bin'));else if(inputs[i].name==='y')inputs[i].setData(this.read('bn_dy.bin'));else throw new Error('Unknown BN diagnostic input '+inputs[i].name);}
  for(let step=0;step<3;step++)if(!model.runStep(inputs))throw new Error('BN diagnostic failed');
  const ws=model.getWeights();for(let i=0;i<ws.length;i++){if(ws[i].name==='gamma'||ws[i].name==='beta'||ws[i].name==='moving_mean'||ws[i].name==='moving_variance')this.save('bn_'+ws[i].name+'.bin',ws[i].getData());}
  this.save(this.c.id+'_diagnostic.json',JSON.stringify({completed:true}));this.message='BatchNorm diagnostic completed';this.log('BN_DIAG_DONE');
 }
 async probeBackend(): Promise<void> {
  const c=this.c;const result=new BackendProbe();result.id=c.id;result.requestedBackend=c.backend;result.training=c.probeTrain;
  result.capabilities=batchprep.nativeCall('backendInfo',0) as string;result.librariesBefore=batchprep.nativeCall('backendLibraries',0) as string;
  this.message='Backend probe: '+c.backend;this.detail=result.capabilities;result.state='building';this.save(c.id+'_backend.json',JSON.stringify(result));this.log('BACKEND_BUILD '+c.backend+' training='+c.probeTrain);await this.yieldUI();
  let loaded: NativeModel|null=null;
  try {
   const model=await NativeModel.load(this.read(c.probeModel),c.probeTrain,c.backend);loaded=model;result.built=true;result.state='built';this.save(c.id+'_backend.json',JSON.stringify(result));
   let before=new Float32Array(0);if(c.probeTrain){model.trainMode=true;this.setLR(model,c.learningRate);before=this.snapshot(model,'probe_before');}
   const inputs=this.fill(model,this.read(c.dataset+'_train_images.bin'),this.read(c.dataset+'_train_labels.bin'),this.read(c.modelId+'_epoch_1.bin'),0,true,c.batchSize);
   for(let step=0;step<c.probeSteps;step++){const start=batchprep.now();if(c.probeTrain)model.runStep(inputs);else await model.predict(inputs);result.stepMs.push(batchprep.now()-start);result.steps++;await this.yieldUI();}
   if(c.probeTrain){const after=this.snapshot(model,'probe_after');for(let i=0;i<after.length;i++){if(!Number.isFinite(after[i]))throw new Error('Nonfinite updated weight');if(after[i]!==before[i])result.changedWeights++;}}
   result.state='completed';result.librariesAfter=batchprep.nativeCall('backendLibraries',0) as string;
  }catch(err){result.state='error';result.error=String(err);result.librariesAfter=batchprep.nativeCall('backendLibraries',0) as string;}
  finally{if(loaded!==null)loaded.destroy();}
  result.elapsedMs=batchprep.now()-this.clockStart;this.displayedElapsed=result.elapsedMs;this.save(c.id+'_backend.json',JSON.stringify(result));this.log('BACKEND_RESULT '+JSON.stringify(result));this.message='Backend probe '+result.state;this.detail=c.backend+' · '+result.steps+' steps\n'+result.error+'\n'+result.capabilities;
 }
 async run(): Promise<void> {
  const report=new Report();
  try {
   this.dir=(this.getUIContext().getHostContext() as common.UIAbilityContext).filesDir;if(!fileIo.accessSync(this.dir+'/job.json'))return;
   this.c=JSON.parse(fileIo.readTextSync(this.dir+'/job.json')) as Config;const c=this.c;report.id=c.id;
   if(c.action!=='train'&&c.action!=='smoke'&&c.action!=='reload'&&c.action!=='reload_smoke'&&c.action!=='diagnose'&&c.action!=='diagnose_bn'&&c.action!=='probe_backend')throw new Error('Unknown action');
   this.vars=JSON.parse(fileIo.readTextSync(this.dir+'/'+c.modelId+'_variables.json')) as Variables;
   this.clockStart=batchprep.now();this.timerId=setInterval(()=>this.updateTimer(),250);
   if(c.action==='probe_backend')await this.probeBackend();else if(c.action==='diagnose_bn')await this.diagnoseBN();else if(c.action==='diagnose')await this.diagnoseOptimizer();else if(c.action==='reload'||c.action==='reload_smoke')await this.reload();else await this.train(report);
  }catch(err){report.state='error';report.error=String(err)+' '+(err as Error).stack;this.message='Training error';this.detail=report.error;this.log('ERROR '+report.error);if(this.dir.length>0)this.save(this.c.id+'_result.json',JSON.stringify(report));}
  finally{if(this.timerId>=0){clearInterval(this.timerId);this.updateTimer();}window.getLastWindow(this.getUIContext().getHostContext() as common.UIAbilityContext).then((mainWindow: window.Window)=>mainWindow.setWindowKeepScreenOn(false)).catch(()=>this.log('Could not release keep-screen-on'));}
 }
 build(){Column({space:24}){Text(this.message).fontSize(28).fontWeight(FontWeight.Bold);Text(this.timer).fontSize(48);Text(this.detail).fontSize(20).lineHeight(30)}.width('100%').height('100%').justifyContent(FlexAlign.Center).padding(36)}
}
