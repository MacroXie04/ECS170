import batchprep from 'libbatchprep.so';
import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { common } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { fileIo } from '@kit.CoreFileKit';
class Config {
 id: string = ''; dataset: string = ''; height: number = 28; width: number = 28; inputChannels: number = 1;
 classes: number = 10; batchSize: number = 50; epochs: number = 10; learningRate: number = 0.03;
 trainCount: number = 0; testCount: number = 0; mean: number[] = []; std: number[] = []; seed: number = 170;
 probe: boolean = false; augmentation: boolean = false; verify: boolean = false;
}
class Metric { epoch: number = 0; loss: number = 0; accuracy: number = 0; learningRate: number = 0; elapsedMs: number = 0; }
class Eval { loss: number = 0; accuracy: number = 0; logits: Float32Array = new Float32Array(0); }
class WeightMeta { name: string = ''; offset: number = 0; count: number = 0; shape: number[] = []; }
class Report {
 id: string = ''; state: string = 'running'; trainCount: number = 0; testCount: number = 0; steps: number = 0;
 epochs: Metric[] = []; testLoss: number = 0; testAccuracy: number = 0; changedWeights: number = 0;
 reloadError: number = 0; elapsedMs: number = 0; error: string = '';
}
@Entry
@Component
struct Index {
 @State message: string = 'Stage 3 — HarmonyOS local training';
 @State timer: string = '00:00:00';
 @State detail: string = 'Waiting for experiment configuration';
 private dir: string = '';
 private c: Config = new Config();
 private seed: number = 170;
 private started: boolean = false;
 private timerId: number = -1;
 private clockStart: number = 0;
 updateTimer(): void { const total=Math.floor((batchprep.now()-this.clockStart)/1000);this.timer=Math.floor(total/3600).toString().padStart(2,'0')+':'+Math.floor(total/60%60).toString().padStart(2,'0')+':'+(total%60).toString().padStart(2,'0'); }
 aboutToDisappear(): void { if(this.timerId>=0)clearInterval(this.timerId); }
 private nativeOrder: Int32Array = new Int32Array(0);
 private emptyOrder: Int32Array = new Int32Array(0);
 private verifiedPlain: boolean = false;
 private verifiedAugmented: boolean = false;
 private imageBatch: Float32Array = new Float32Array(0);
 private labelBatch: Float32Array = new Float32Array(0);
 aboutToAppear(): void { if (!this.started) { this.started = true; this.run(); } }
 log(s: string): void { hilog.info(0x170, 'Stage3Train', '%{public}s', this.c.id + ' ' + s); }
 save(n: string, data: ArrayBuffer | string): void {
  const f=fileIo.openSync(this.dir+'/'+n,fileIo.OpenMode.CREATE|fileIo.OpenMode.READ_WRITE|fileIo.OpenMode.TRUNC);
  fileIo.writeSync(f.fd,data);fileIo.closeSync(f);
 }
 read(n: string): ArrayBuffer {
  const path=this.dir+'/'+n;const data=new ArrayBuffer(fileIo.statSync(path).size);
  const f=fileIo.openSync(path,fileIo.OpenMode.READ_ONLY);const count=fileIo.readSync(f.fd,data);fileIo.closeSync(f);
  if(count!==data.byteLength) throw new Error('Incomplete read '+n);return data;
 }
 random(): number {this.seed=(Math.imul(1664525,this.seed)+1013904223)>>>0;return this.seed/4294967296;}
 legacyFill(inputs: mindSporeLite.MSTensor[], images: Uint8Array, labels: Uint8Array, order: number[], start: number, augment: boolean): void {
  const c=this.c;const h=c.height;const w=c.width;const channels=c.inputChannels;const area=h*w;
  this.labelBatch.fill(0);
  for(let b=0;b<c.batchSize;b++) {
   const idx=order.length>0?order[start+b]:start+b;
   this.labelBatch[b*c.classes+labels[idx]]=1;
   const flip=augment&&this.random()<0.5;
   const dy=augment?Math.floor(this.random()*9)-4:0;const dx=augment?Math.floor(this.random()*9)-4:0;
   for(let ch=0;ch<channels;ch++) for(let y=0;y<h;y++) for(let x=0;x<w;x++) {
    const sy=y+dy;const sx=(flip?w-1-x:x)+dx;
    const v=(sy>=0&&sy<h&&sx>=0&&sx<w)?images[(idx*area+sy*w+sx)*channels+ch]/255:0;
    this.imageBatch[(b*channels+ch)*area+y*w+x]=(v-c.mean[ch])/c.std[ch];
   }
  }
  for(let i=0;i<inputs.length;i++) {
   if(inputs[i].elementNum===this.imageBatch.length) inputs[i].setData(this.imageBatch.buffer as ArrayBuffer);
   else if(inputs[i].elementNum===this.labelBatch.length) inputs[i].setData(this.labelBatch.buffer as ArrayBuffer);
   else throw new Error('Unexpected tensor '+inputs[i].name);
  }
 }
 fill(inputs: mindSporeLite.MSTensor[], images: Uint8Array, labels: Uint8Array, order: number[], start: number, augment: boolean): void {
  const verify=augment?!this.verifiedAugmented:!this.verifiedPlain;
  const originalSeed=this.seed;
  let expected=new Float32Array(0);let expectedLabels=new Float32Array(0);let expectedSeed=this.seed;
  if(verify){this.legacyFill(inputs,images,labels,order,start,augment);expected=this.imageBatch.slice();expectedLabels=this.labelBatch.slice();expectedSeed=this.seed;this.seed=originalSeed;}
  this.seed=batchprep.fill(images.buffer as ArrayBuffer,labels.buffer as ArrayBuffer,(order.length>0?this.nativeOrder.buffer:this.emptyOrder.buffer) as ArrayBuffer,this.imageBatch.buffer as ArrayBuffer,this.labelBatch.buffer as ArrayBuffer,start,this.seed,augment);
  if(verify){
   for(let i=0;i<expected.length;i++)if(expected[i]!==this.imageBatch[i])throw new Error('Native image preprocessing mismatch');
   for(let i=0;i<expectedLabels.length;i++)if(expectedLabels[i]!==this.labelBatch[i])throw new Error('Native label preprocessing mismatch');
   if(expectedSeed!==this.seed)throw new Error('Native RNG mismatch');
   if(augment)this.verifiedAugmented=true;else this.verifiedPlain=true;
   this.log('PREPROCESS_EQUIVALENCE_PASS augmentation='+augment);
  }
  for(let i=0;i<inputs.length;i++){
   if(inputs[i].elementNum===this.imageBatch.length)inputs[i].setData(this.imageBatch.buffer as ArrayBuffer);
   else if(inputs[i].elementNum===this.labelBatch.length)inputs[i].setData(this.labelBatch.buffer as ArrayBuffer);
   else throw new Error('Unexpected native batch tensor');
  }
 }
 async evaluate(model: mindSporeLite.Model, images: Uint8Array, labels: Uint8Array): Promise<Eval> {
  const r=new Eval();const c=this.c;r.logits=new Float32Array(labels.length*c.classes);let correct=0;
  const inputs=model.getInputs();
  for(let start=0;start<labels.length;start+=c.batchSize) {
   this.fill(inputs,images,labels,[],start,false);const out=await model.predict(inputs);
   const values=new Float32Array(out[0].getData());
   if(values.length!==c.batchSize*c.classes) throw new Error('Unexpected logits shape');
   r.logits.set(values,start*c.classes);
   for(let b=0;b<c.batchSize;b++) {
    let peak=-Infinity;let arg=0;
    for(let j=0;j<c.classes;j++){const v=values[b*c.classes+j];if(!Number.isFinite(v)) throw new Error('Non-finite logits');if(v>peak){peak=v;arg=j;}}
    let sum=0;for(let j=0;j<c.classes;j++)sum+=Math.exp(values[b*c.classes+j]-peak);
    r.loss+=Math.log(sum)+peak-values[b*c.classes+labels[start+b]];if(arg===labels[start+b])correct++;
   }
   if(start%(c.batchSize*20)===0) await new Promise<void>((resolve)=>setTimeout(resolve,1));
  }
  r.loss/=labels.length;r.accuracy=correct/labels.length;return r;
 }
 snapshot(model: mindSporeLite.Model, suffix: string): Float32Array {
  const ts=model.getWeights();const meta: WeightMeta[]=[];let total=0;
  for(let i=0;i<ts.length;i++) {
   const name=ts[i].name;
   if((name.endsWith('.weight')||name.endsWith('.bias'))&&!name.startsWith('accum.')&&!name.startsWith('stat.')) {
    const t=new WeightMeta();t.name=name;t.shape=ts[i].shape;t.offset=total;t.count=ts[i].elementNum;total+=t.count;meta.push(t);
   }
  }
  const flat=new Float32Array(total);let offset=0;
  for(let i=0;i<ts.length;i++) {
   const name=ts[i].name;
   if((name.endsWith('.weight')||name.endsWith('.bias'))&&!name.startsWith('accum.')&&!name.startsWith('stat.')) {
    const values=new Float32Array(ts[i].getData());flat.set(values,offset);offset+=values.length;
   }
  }
  this.save(this.c.id+'_'+suffix+'.bin',flat.buffer as ArrayBuffer);this.save(this.c.id+'_'+suffix+'.json',JSON.stringify(meta));return flat;
 }
 async probe(): Promise<void> {
  const started=batchprep.now();this.clockStart=started;this.timerId=setInterval(()=>this.updateTimer(),250);
  this.message='PR #6 full CNN — device compatibility probe';
  const raw=this.read(this.c.id+'.ms');
  const model=await mindSporeLite.loadTrainModelFromBuffer(raw,{lossName:['loss'],optimizationLevel:mindSporeLite.OptimizationLevel.O0},{target:['cpu'],cpu:{threadNum:4,precisionMode:'enforce_fp32'}});
  model.trainMode=true;
  const inputs=model.getInputs();
  if(!inputs || inputs.length===0)throw new Error('Training graph rejected by device runtime');
  const images=new Uint8Array(this.read('probe_images.bin'));const labels=new Uint8Array(this.read('probe_labels.bin'));
  const before=this.snapshot(model,'weights_before');
  this.fill(inputs,images,labels,[],0,false);
  const stepStarted=batchprep.now();
  if(!model.runStep(inputs))throw new Error('PR6 compatibility training step failed');
  const trainMs=batchprep.now()-stepStarted;this.updateTimer();
  const after=this.snapshot(model,'weights_after');let changed=0;
  for(let i=0;i<before.length;i++)if(before[i]!==after[i])changed++;
  if(changed===0)throw new Error('No parameter updates in compatibility probe');
  this.save(this.c.id+'_probe.json',JSON.stringify({status:'probe_passed',formalTraining:false,optimizerSteps:1,trainMs:trainMs,totalMs:batchprep.now()-started,changedWeights:changed}));
  this.message='PR #6 graph compatibility: PASS';this.detail='One real-data optimizer step: '+(trainMs/1000).toFixed(3)+' seconds. Full experiment has not started.';
 }
 async run(): Promise<void> {
  const report=new Report();
  try {
   const context=this.getUIContext().getHostContext() as common.UIAbilityContext;this.dir=context.filesDir;
   if(!fileIo.accessSync(this.dir+'/job.json'))return;
   this.c=JSON.parse(fileIo.readTextSync(this.dir+'/job.json')) as Config;const c=this.c;
   this.seed=c.seed;batchprep.configure(c.height,c.width,c.inputChannels,c.classes,c.batchSize,new Float64Array(c.mean).buffer as ArrayBuffer,new Float64Array(c.std).buffer as ArrayBuffer);report.id=c.id;report.trainCount=c.trainCount;report.testCount=c.testCount;
   this.imageBatch=new Float32Array(c.batchSize*c.height*c.width*c.inputChannels);this.labelBatch=new Float32Array(c.batchSize*c.classes);
   if(c.probe){await this.probe();return;}
   throw new Error('Formal PR6 training is not implemented yet; legacy jobs must use the archived runner.');
  } catch(err) {
   report.state='error';report.error=String(err)+' '+JSON.stringify(err)+' '+(err as Error).stack;
   this.message='Training error';this.detail=report.error;this.log('ERROR '+report.error);
   if(this.dir.length>0)this.save(this.c.id+'_result.json',JSON.stringify(report));
  } finally { if(this.timerId>=0){clearInterval(this.timerId);this.updateTimer();} }
 }
 build(){Column({space:24}){Text(this.message).fontSize(30).fontWeight(FontWeight.Bold);Text(this.timer).fontSize(28);Text(this.detail).fontSize(20).lineHeight(32)}.width('100%').height('100%').justifyContent(FlexAlign.Center).padding(40)}
}
