const EPOCH = Date.UTC(2026, 0, 1);
const WORKER_KEY = "chat-worker-id";

let lastTimestamp = -1n;
let sequence = 0n;
let worker: bigint | null = null;

function workerId(): bigint {
  if (worker !== null) return worker;
  try {
    const stored = localStorage.getItem(WORKER_KEY);
    if (stored && /^\d{1,4}$/.test(stored)) {
      const value = BigInt(stored);
      if (value >= 0n && value < 1024n) {
        worker = value;
        return value;
      }
    }
  } catch {
    // localStorage may be unavailable
  }
  worker = BigInt(Math.floor(Math.random() * 1024));
  try {
    localStorage.setItem(WORKER_KEY, worker.toString());
  } catch {
    // localStorage may be unavailable
  }
  return worker;
}

export function createSessionId(): string {
  let timestamp = BigInt(Math.max(0, Date.now() - EPOCH));
  if (timestamp === lastTimestamp) {
    sequence = (sequence + 1n) & 4095n;
    if (sequence === 0n) {
      do {
        timestamp = BigInt(Math.max(0, Date.now() - EPOCH));
      } while (timestamp === lastTimestamp);
    }
  } else {
    sequence = 0n;
  }
  lastTimestamp = timestamp;
  return ((timestamp << 22n) | (workerId() << 12n) | sequence).toString();
}
