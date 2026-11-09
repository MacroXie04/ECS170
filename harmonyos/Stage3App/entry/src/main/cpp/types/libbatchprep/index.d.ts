export const configure: (height: number, width: number, channels: number, classes: number, batch: number, mean: ArrayBuffer, std: ArrayBuffer) => void;
export const fill: (images: ArrayBuffer, labels: ArrayBuffer, order: ArrayBuffer, x: ArrayBuffer, y: ArrayBuffer, start: number, seed: number, augmentation: boolean) => number;

export const now: () => number;

export const fillScheduled: (images: ArrayBuffer, labels: ArrayBuffer, schedule: ArrayBuffer, x: ArrayBuffer, y: ArrayBuffer, start: number, replicate: boolean, training: boolean) => void;

export const nativeCall: (op: string, arg: number | ArrayBuffer, value?: boolean | number | string | ArrayBuffer | number[], infer?: boolean | string) => number | boolean | string | ArrayBuffer | TensorInfo[];
export const mnnAsync: (op: string, a?: string, b?: string, c?: string) => Promise<string>;
export interface TensorInfo { handle: number; name: string; shape: number[]; elementNum: number; }
