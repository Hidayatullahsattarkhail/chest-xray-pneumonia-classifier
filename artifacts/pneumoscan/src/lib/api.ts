export interface PredictionResult {
  label: "NORMAL" | "PNEUMONIA";
  confidences: {
    label: "NORMAL" | "PNEUMONIA";
    confidence: number;
  }[];
}

const GRADIO_API_URL = "https://hidayatullah099-pneumonia-xray-classifier.hf.space/gradio_api/call/predict";

export async function analyzeImage(
  base64DataUri: string,
  onStatusUpdate: (status: string) => void,
  signal: AbortSignal
): Promise<PredictionResult> {
  try {
    onStatusUpdate("Initiating analysis...");

    // Step 1: POST to start prediction
    const postResponse = await fetch(GRADIO_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: [{ path: base64DataUri, meta: { _type: "gradio.FileData" } }],
      }),
      signal,
    });

    if (!postResponse.ok) {
      throw new Error(`Failed to start prediction: ${postResponse.status} ${postResponse.statusText}`);
    }

    const { event_id } = await postResponse.json();
    if (!event_id) throw new Error("No event_id returned from API");

    console.log("[PneumoScan] event_id:", event_id);
    onStatusUpdate("AI model is processing... (may take up to 30 seconds on cold start)");

    // Step 2: GET SSE stream
    const sseResponse = await fetch(`${GRADIO_API_URL}/${event_id}`, {
      method: "GET",
      signal,
    });

    if (!sseResponse.ok) {
      throw new Error(`SSE stream error: ${sseResponse.status} ${sseResponse.statusText}`);
    }

    if (!sseResponse.body) throw new Error("Response body is empty");

    const reader = sseResponse.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let lastValidResult: PredictionResult | null = null;
    let lastEventType = "";

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        // Process any remaining data in buffer
        if (buffer.trim()) {
          console.log("[PneumoScan] final buffer line:", buffer);
          tryParseLine(buffer, lastEventType, (r) => { lastValidResult = r; });
        }
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Keep the last (possibly partial) line in buffer
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        // Log every raw line
        console.log("[PneumoScan] SSE line:", JSON.stringify(line));

        if (line.startsWith("event: ")) {
          lastEventType = line.slice(7).trim();
          console.log("[PneumoScan] event type:", lastEventType);
        } else if (line.startsWith("data: ")) {
          const dataStr = line.slice(6).trim();
          tryParseLine(dataStr, lastEventType, (r) => { lastValidResult = r; });
        }
      }
    }

    if (lastValidResult) return lastValidResult;
    throw new Error("Stream ended without receiving a valid result.");
  } catch (error: unknown) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Request timed out after 60 seconds.");
    }
    throw error;
  }
}

function tryParseLine(
  dataStr: string,
  eventType: string,
  onResult: (r: PredictionResult) => void
) {
  try {
    const parsed = JSON.parse(dataStr);
    console.log("[PneumoScan] parsed data (event=" + eventType + "):", parsed);

    // Case 1: array where first element has label + confidences
    if (Array.isArray(parsed) && parsed.length > 0) {
      const candidate = parsed[0];
      if (candidate && typeof candidate === "object" && "label" in candidate && "confidences" in candidate) {
        console.log("[PneumoScan] ✓ valid result found:", candidate);
        onResult(candidate as PredictionResult);
        return;
      }
    }

    // Case 2: direct object with label + confidences
    if (parsed && typeof parsed === "object" && "label" in parsed && "confidences" in parsed) {
      console.log("[PneumoScan] ✓ valid result (direct object):", parsed);
      onResult(parsed as PredictionResult);
      return;
    }

    // Case 3: nested in output.data
    if (parsed?.output?.data?.[0]?.label) {
      const candidate = parsed.output.data[0];
      console.log("[PneumoScan] ✓ valid result (nested output.data):", candidate);
      onResult(candidate as PredictionResult);
      return;
    }

    console.log("[PneumoScan] data line has no label field, skipping");
  } catch {
    console.log("[PneumoScan] could not parse data line:", dataStr);
  }
}
