export interface PredictionResult {
  label: "NORMAL" | "PNEUMONIA";
  confidences: {
    label: "NORMAL" | "PNEUMONIA";
    confidence: number;
  }[];
}

const BASE = "https://hidayatullah099-pneumonia-xray-classifier.hf.space";
const UPLOAD_URL = `${BASE}/gradio_api/upload`;
const PREDICT_URL = `${BASE}/gradio_api/call/predict`;

/**
 * Step 1: Upload the raw file via multipart/form-data.
 * Gradio 4.x returns an array of server-side file paths.
 */
async function uploadFile(file: File, signal: AbortSignal): Promise<string> {
  const form = new FormData();
  form.append("files", file);

  const res = await fetch(UPLOAD_URL, { method: "POST", body: form, signal });
  if (!res.ok) throw new Error(`Upload failed: ${res.status} ${res.statusText}`);

  const paths: string[] = await res.json();
  console.log("[PneumoScan] uploaded file paths:", paths);

  if (!Array.isArray(paths) || paths.length === 0 || !paths[0]) {
    throw new Error("Upload returned no file path");
  }
  return paths[0];
}

/**
 * Step 2: Start the prediction job using the server-side path.
 * Returns the event_id to poll.
 */
async function startPredict(serverPath: string, signal: AbortSignal): Promise<string> {
  const body = {
    data: [{ path: serverPath, meta: { _type: "gradio.FileData" } }],
  };

  const res = await fetch(PREDICT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) throw new Error(`Predict start failed: ${res.status} ${res.statusText}`);

  const json = await res.json();
  console.log("[PneumoScan] predict POST response:", json);

  const event_id = json?.event_id;
  if (!event_id) throw new Error("No event_id in predict response");
  return event_id;
}

/**
 * Step 3: Read the SSE stream and return the last valid PredictionResult.
 */
async function readSSEResult(
  event_id: string,
  onStatusUpdate: (s: string) => void,
  signal: AbortSignal
): Promise<PredictionResult> {
  const sseRes = await fetch(`${PREDICT_URL}/${event_id}`, { method: "GET", signal });
  if (!sseRes.ok) throw new Error(`SSE stream error: ${sseRes.status} ${sseRes.statusText}`);
  if (!sseRes.body) throw new Error("SSE response body is empty");

  const reader = sseRes.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let lastValidResult: PredictionResult | null = null;
  let lastEventType = "";

  while (true) {
    const { done, value } = await reader.read();

    if (done) {
      if (buffer.trim()) {
        console.log("[PneumoScan] final buffer:", buffer);
        parseLine(buffer, lastEventType, (r) => { lastValidResult = r; });
      }
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      console.log("[PneumoScan] SSE line:", JSON.stringify(line));

      if (line.startsWith("event: ")) {
        lastEventType = line.slice(7).trim();
        console.log("[PneumoScan] event type:", lastEventType);

        if (lastEventType === "error") {
          onStatusUpdate("Model returned an error — retrying may help if it just woke up...");
        } else if (lastEventType === "generating") {
          onStatusUpdate("Generating prediction...");
        } else if (lastEventType === "complete") {
          onStatusUpdate("Finalising result...");
        }
      } else if (line.startsWith("data: ")) {
        const dataStr = line.slice(6).trim();
        parseLine(dataStr, lastEventType, (r) => { lastValidResult = r; });

        // If we hit an error event with error content, surface it
        if (lastEventType === "error") {
          try {
            const parsed = JSON.parse(dataStr);
            if (parsed?.error && typeof parsed.error === "string") {
              throw new Error(`Model error: ${parsed.error}`);
            }
          } catch (e: unknown) {
            if (e instanceof Error && e.message.startsWith("Model error:")) throw e;
          }
        }
      }
    }
  }

  if (lastValidResult) return lastValidResult;
  throw new Error("Stream ended without a valid prediction. The model may still be warming up — please try again in 30 seconds.");
}

function parseLine(
  dataStr: string,
  eventType: string,
  onResult: (r: PredictionResult) => void
) {
  try {
    const parsed = JSON.parse(dataStr);
    console.log(`[PneumoScan] parsed data (event=${eventType}):`, parsed);

    // Shape A: [{label, confidences}]
    if (Array.isArray(parsed) && parsed.length > 0) {
      const c = parsed[0];
      if (c && typeof c === "object" && "label" in c && "confidences" in c) {
        console.log("[PneumoScan] valid result (shape A):", c);
        onResult(c as PredictionResult);
        return;
      }
    }

    // Shape B: {label, confidences} directly
    if (parsed && typeof parsed === "object" && "label" in parsed && "confidences" in parsed) {
      console.log("[PneumoScan] valid result (shape B):", parsed);
      onResult(parsed as PredictionResult);
      return;
    }

    // Shape C: {output: {data: [{label, confidences}]}}
    const nested = parsed?.output?.data?.[0];
    if (nested && "label" in nested && "confidences" in nested) {
      console.log("[PneumoScan] valid result (shape C):", nested);
      onResult(nested as PredictionResult);
      return;
    }

    console.log("[PneumoScan] no label field — skipping");
  } catch {
    console.log("[PneumoScan] could not parse line:", dataStr);
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

export async function analyzeImage(
  file: File,
  onStatusUpdate: (status: string) => void,
  signal: AbortSignal
): Promise<PredictionResult> {
  try {
    onStatusUpdate("Uploading image to AI server...");
    const serverPath = await uploadFile(file, signal);

    onStatusUpdate("Image uploaded. Starting AI analysis...");
    const event_id = await startPredict(serverPath, signal);
    console.log("[PneumoScan] event_id:", event_id);

    onStatusUpdate("AI model is processing... (may take up to 30 seconds on first use)");
    return await readSSEResult(event_id, onStatusUpdate, signal);
  } catch (err: unknown) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("Request timed out after 60 seconds. The model may be warming up — please try again.");
    }
    throw err;
  }
}
