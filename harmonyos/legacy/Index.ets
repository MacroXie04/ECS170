import batchprep from 'libbatchprep.so';
import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { common } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { fileIo } from '@kit.CoreFileKit';
class Config {
 id: string = ''; dataset: string = ''; height: number = 28; width: number = 28; inputChannels: number = 1;
 classes: number = 10; batchSize: number = 50; epochs: number = 10; learningRate: number = 0.03;
 trainCount: number = 0; testCount: number = 0; mean: number[] = []; std: number[] = []; seed: number = 170;
 augmentation: boolean = false; verify: boolean = false;
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
 @State detail: string = 'Waiting for experiment configuration';
 private dir: string = '';
 private c: Config = new Config();
 private seed: number = 170;
 private started: boolean = false;
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
 async run(): Promise<void> {
  const report=new Report();
  try {
   const context=this.getUIContext().getHostContext() as common.UIAbilityContext;this.dir=context.filesDir;
   if(!fileIo.accessSync(this.dir+'/job.json'))return;
   this.c=JSON.parse(fileIo.readTextSync(this.dir+'/job.json')) as Config;const c=this.c;
   this.seed=c.seed;batchprep.configure(c.height,c.width,c.inputChannels,c.classes,c.batchSize,new Float64Array(c.mean).buffer as ArrayBuffer,new Float64Array(c.std).buffer as ArrayBuffer);report.id=c.id;report.trainCount=c.trainCount;report.testCount=c.testCount;
   this.imageBatch=new Float32Array(c.batchSize*c.height*c.width*c.inputChannels);this.labelBatch=new Float32Array(c.batchSize*c.classes);
   const testImages=new Uint8Array(this.read(c.dataset+'_test_images.bin'));const testLabels=new Uint8Array(this.read(c.dataset+'_test_labels.bin'));
   if(c.verify) {
    const saved=await mindSporeLite.loadModelFromFile(this.dir+'/'+c.id+'_inference.ms',{target:['cpu'],cpu:{threadNum:2,precisionMode:'enforce_fp32'}});
    const result=await this.evaluate(saved,testImages,testLabels);const expected=new Float32Array(this.read(c.id+'_test_logits.bin'));let delta=0;
    for(let i=0;i<expected.length;i++)delta=Math.max(delta,Math.abs(expected[i]-result.logits[i]));
    this.save(c.id+'_reloaded_logits.bin',result.logits.buffer as ArrayBuffer);
    this.save(c.id+'_reload.json',JSON.stringify({passed:delta<0.0001,maxError:delta,testLoss:result.loss,testAccuracy:result.accuracy}));
    this.message='Saved model reload '+(delta<0.0001?'PASS':'FAIL');this.detail=c.id+' · test accuracy '+(100*result.accuracy).toFixed(2)+'%';this.log('RELOAD '+delta);return;
   }
   const images=new Uint8Array(this.read(c.dataset+'_train_images.bin'));const labels=new Uint8Array(this.read(c.dataset+'_train_labels.bin'));
   if(labels.length!==c.trainCount||testLabels.length!==c.testCount)throw new Error('Dataset count mismatch');
   const raw=this.read(c.id+'.ms');
   const model=await mindSporeLite.loadTrainModelFromBuffer(raw,{lossName:['loss'],optimizationLevel:mindSporeLite.OptimizationLevel.O0},{target:['cpu'],cpu:{threadNum:2,precisionMode:'enforce_fp32'}});
   model.trainMode=true;if(!model.trainMode)throw new Error('Training mode not enabled');
   const before=this.snapshot(model,'weights_before');const inputs=model.getInputs();
   const order:number[]=[];for(let i=0;i<labels.length;i++)order.push(i);
   const startTime=Date.now();
   for(let epoch=0;epoch<=c.epochs;epoch++) {
    const lr=c.learningRate*(epoch>Math.floor(c.epochs*0.8)?0.1:epoch>Math.floor(c.epochs*0.55)?0.3:1);
    if(epoch>0) {
     model.learningRate=lr;
     for(let i=order.length-1;i>0;i--){const j=Math.floor(this.random()*(i+1));const t=order[i];order[i]=order[j];order[j]=t;}
     this.nativeOrder=new Int32Array(order);
     for(let start=0;start<labels.length;start+=c.batchSize) {
      this.fill(inputs,images,labels,order,start,c.augmentation);
      if(!model.runStep(inputs))throw new Error('runStep failed');report.steps++;
      if(start%(c.batchSize*20)===0){this.message=c.id+' epoch '+epoch+'/'+c.epochs;this.detail='Training '+start+'/'+labels.length+' images';await new Promise<void>((resolve)=>setTimeout(resolve,1));}
     }
    }
    const clone=await mindSporeLite.loadTrainModelFromBuffer(raw,{lossName:['loss'],optimizationLevel:mindSporeLite.OptimizationLevel.O0},{target:['cpu'],cpu:{threadNum:2}});
    if(!clone.updateWeights(model.getWeights()))throw new Error('Weight copy failed');
    if(!clone.exportModel(this.dir+'/'+c.id+'_epoch.ms',mindSporeLite.QuantizationType.NO_QUANT,true))throw new Error('Export failed');
    const evaluator=await mindSporeLite.loadModelFromFile(this.dir+'/'+c.id+'_epoch.ms',{target:['cpu'],cpu:{threadNum:2,precisionMode:'enforce_fp32'}});
    const e=await this.evaluate(evaluator,images,labels);const metric=new Metric();metric.epoch=epoch;metric.loss=e.loss;metric.accuracy=e.accuracy;metric.learningRate=lr;metric.elapsedMs=Date.now()-startTime;report.epochs.push(metric);
    this.save(c.id+'_progress.json',JSON.stringify(report));this.log('EPOCH '+JSON.stringify(metric));
    if(epoch===c.epochs) {
     const t=await this.evaluate(evaluator,testImages,testLabels);report.testLoss=t.loss;report.testAccuracy=t.accuracy;this.save(c.id+'_test_logits.bin',t.logits.buffer as ArrayBuffer);
    }
   }
   const after=this.snapshot(model,'weights_after');for(let i=0;i<before.length;i++)if(before[i]!==after[i])report.changedWeights++;
   if(!model.exportModel(this.dir+'/'+c.id+'_checkpoint.ms',mindSporeLite.QuantizationType.NO_QUANT,false))throw new Error('Checkpoint save failed');
   if(!model.exportModel(this.dir+'/'+c.id+'_inference.ms',mindSporeLite.QuantizationType.NO_QUANT,true))throw new Error('Inference save failed');
   report.elapsedMs=Date.now()-startTime;report.state=(report.changedWeights>0&&report.epochs[c.epochs].loss<report.epochs[0].loss)?'completed':'failed';
   this.save(c.id+'_result.json',JSON.stringify(report));this.log('COMPLETE accuracy='+report.testAccuracy+' state='+report.state);
   this.message=c.id+' '+report.state;this.detail='All '+c.trainCount+' training images · '+c.epochs+' epochs\nTest accuracy '+(100*report.testAccuracy).toFixed(2)+'%\nModel saved locally';
  } catch(err) {
   report.state='error';report.error=String(err)+' '+JSON.stringify(err)+' '+(err as Error).stack;
   this.message='Training error';this.detail=report.error;this.log('ERROR '+report.error);
   if(this.dir.length>0)this.save(this.c.id+'_result.json',JSON.stringify(report));
  }
 }
 build(){Column({space:24}){Text(this.message).fontSize(30).fontWeight(FontWeight.Bold);Text(this.detail).fontSize(20).lineHeight(32)}.width('100%').height('100%').justifyContent(FlexAlign.Center).padding(40)}
}
