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
    
    // Step 1: Start prediction (POST)
    const postResponse = await fetch(GRADIO_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data: [{ path: base64DataUri, meta: { _type: "gradio.FileData" } }]
      }),
      signal,
    });

    if (!postResponse.ok) {
      throw new Error(`Failed to start prediction: ${postResponse.statusText}`);
    }

    const { event_id } = await postResponse.json();
    
    if (!event_id) {
      throw new Error("No event_id returned from API");
    }

    onStatusUpdate("AI model is processing... (This may take up to 30 seconds if warming up)");

    // Step 2: Get result via SSE stream (GET)
    const sseResponse = await fetch(`${GRADIO_API_URL}/${event_id}`, {
      method: "GET",
      signal,
    });

    if (!sseResponse.ok) {
      throw new Error(`Failed to read prediction stream: ${sseResponse.statusText}`);
    }

    if (!sseResponse.body) {
      throw new Error("Response body is empty");
    }

    const reader = sseResponse.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      
      // Keep the last partial line in the buffer
      buffer = lines.pop() || "";
      
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const dataStr = line.substring(6).trim();
          
          try {
            const data = JSON.parse(dataStr);
            // The result is usually an array, first element is the output object
            if (Array.isArray(data) && data.length > 0 && data[0].label && data[0].confidences) {
              return data[0] as PredictionResult;
            } else if (data.msg === "process_completed" && data.output?.data?.[0]) {
               return data.output.data[0] as PredictionResult;
            }
          } catch (e) {
            // Ignore parse errors for partial or status messages, continue reading
            console.log("Ignored SSE data:", dataStr);
          }
        }
      }
    }

    throw new Error("Stream ended without receiving a valid result.");
  } catch (error: any) {
    if (error.name === "AbortError") {
      throw new Error("Request timed out after 60 seconds.");
    }
    throw error;
  }
}
