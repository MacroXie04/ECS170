export const configure: (height: number, width: number, channels: number, classes: number, batch: number, mean: ArrayBuffer, std: ArrayBuffer) => void;
export const fill: (images: ArrayBuffer, labels: ArrayBuffer, order: ArrayBuffer, x: ArrayBuffer, y: ArrayBuffer, start: number, seed: number, augmentation: boolean) => number;

export const now: () => number;
