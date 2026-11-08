import bridge, { TensorInfo } from 'libbatchprep.so';
export class NativeTensor {
 handle: number; name: string; shape: number[]; elementNum: number;
 constructor(info: TensorInfo){this.handle=info.handle;this.name=info.name;this.shape=info.shape;this.elementNum=info.elementNum;}
 getData(): ArrayBuffer {return bridge.nativeCall('data',this.handle) as ArrayBuffer;}
 setData(data: ArrayBuffer): void {bridge.nativeCall('setData',this.handle,data);}
}
export class NativeModel {
 handle: number;
 constructor(data: ArrayBuffer,train: boolean){this.handle=bridge.nativeCall('load',data,train) as number;}
 static async load(data: ArrayBuffer,train: boolean): Promise<NativeModel>{return new NativeModel(data,train);}
 tensors(op: string): NativeTensor[]{const info=bridge.nativeCall(op,this.handle) as TensorInfo[];return info.map((t: TensorInfo)=>new NativeTensor(t));}
 destroy(): void {bridge.nativeCall('destroy',this.handle);}
 getInputs(): NativeTensor[]{return this.tensors('inputs');}
 getWeights(): NativeTensor[]{return this.tensors('weights');}
 updateWeights(weights: NativeTensor[]): boolean {bridge.nativeCall('update',this.handle,weights.map((t: NativeTensor)=>t.handle));return true;}
 runStep(inputs: NativeTensor[]): boolean {bridge.nativeCall('step',this.handle);return true;}
 async predict(inputs: NativeTensor[]): Promise<NativeTensor[]>{return this.tensors('predict');}
 set trainMode(mode: boolean){bridge.nativeCall('mode',this.handle,mode);}
 get trainMode(): boolean{return bridge.nativeCall('mode',this.handle) as boolean;}
 set learningRate(lr: number){bridge.nativeCall('lr',this.handle,lr);}
 get learningRate(): number{return bridge.nativeCall('lr',this.handle) as number;}
 exportModel(path: string,quant: number,infer: boolean): boolean {bridge.nativeCall('export',this.handle,path,infer);return true;}
}
