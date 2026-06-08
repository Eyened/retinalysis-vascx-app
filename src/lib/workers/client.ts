import type { ProgressMessage, WorkerRequest, WorkerRequestMap, WorkerResponse } from '$lib/types/pipeline';

export type ProgressHandler = (message: ProgressMessage) => void;

export class PipelineWorkerClient {
  private worker: Worker;
  private pending = new Map<string, { resolve: (value: unknown) => void; reject: (reason: Error) => void }>();
  private sequence = 0;
  private onProgress?: ProgressHandler;

  constructor(worker: Worker, onProgress?: ProgressHandler) {
    this.worker = worker;
    this.onProgress = onProgress;
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handleMessage(event.data);
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'Worker failed.');
      for (const pending of this.pending.values()) {
        pending.reject(error);
      }
      this.pending.clear();
    };
  }

  request<K extends keyof WorkerRequestMap>(message: WorkerRequestMap[K], transfer: Transferable[] = []): Promise<unknown> {
    const requestId = `${message.type}-${++this.sequence}`;
    const payload = { ...message, requestId } as WorkerRequest;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      this.worker.postMessage(payload, transfer);
    });
  }

  terminate() {
    this.worker.terminate();
    this.pending.clear();
  }

  private handleMessage(message: WorkerResponse) {
    if (message.type === 'progress') {
      this.onProgress?.(message);
      return;
    }

    const pending = this.pending.get(message.requestId);
    if (!pending) {
      return;
    }
    this.pending.delete(message.requestId);

    if (message.type === 'error') {
      pending.reject(new Error(message.error.detail ? `${message.error.message}: ${message.error.detail}` : message.error.message));
      return;
    }

    pending.resolve(message.payload);
  }
}
