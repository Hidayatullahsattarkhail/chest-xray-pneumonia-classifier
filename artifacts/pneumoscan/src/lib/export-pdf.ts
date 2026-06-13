import { jsPDF } from "jspdf";
import type { PredictionResult } from "./api";

const TEAL = [0, 128, 128] as const;
const DARK = [15, 30, 50] as const;
const GRAY = [100, 116, 139] as const;
const LIGHT_GRAY = [241, 245, 249] as const;
const GREEN = [22, 163, 74] as const;
const RED = [220, 38, 38] as const;
const WHITE = [255, 255, 255] as const;

function setColor(doc: jsPDF, rgb: readonly [number, number, number], type: "fill" | "draw" | "text") {
  if (type === "fill") doc.setFillColor(rgb[0], rgb[1], rgb[2]);
  else if (type === "draw") doc.setDrawColor(rgb[0], rgb[1], rgb[2]);
  else doc.setTextColor(rgb[0], rgb[1], rgb[2]);
}

export async function exportReportPDF(
  imageDataUrl: string,
  fileName: string,
  result: PredictionResult
): Promise<void> {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const W = 210;
  let y = 0;

  // ── Header bar ──────────────────────────────────────────────────────────────
  setColor(doc, TEAL, "fill");
  doc.rect(0, 0, W, 28, "F");

  setColor(doc, WHITE, "text");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.text("PneumoScan AI", 14, 12);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.text("CLINICAL DEMO TOOL  •  AI-Powered Chest X-Ray Analysis", 14, 19);

  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  const timeStr = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  doc.text(`Report generated: ${dateStr} at ${timeStr}`, W - 14, 19, { align: "right" });

  y = 36;

  // ── Section: Patient / File Info ────────────────────────────────────────────
  setColor(doc, LIGHT_GRAY, "fill");
  doc.rect(14, y, W - 28, 10, "F");
  setColor(doc, DARK, "text");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.text("ANALYSIS DETAILS", 18, y + 6.5);
  y += 10;

  setColor(doc, DARK, "text");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.text(`File name:`, 18, y + 7);
  doc.setFont("helvetica", "bold");
  doc.text(fileName, 45, y + 7);
  doc.setFont("helvetica", "normal");
  doc.text(`Model:`, 18, y + 14);
  doc.setFont("helvetica", "bold");
  doc.text("DenseNet121 — Hugging Face Space (hidayatullah099/pneumonia-xray-classifier)", 45, y + 14);
  y += 22;

  // ── Two-column layout: image + results ──────────────────────────────────────
  const colGap = 6;
  const imgCol = 80;
  const resCol = W - 28 - imgCol - colGap;
  const imgX = 14;
  const resX = imgX + imgCol + colGap;

  // Load the image to get natural dimensions
  const imgProps = await loadImageProps(doc, imageDataUrl);
  const imgH = (imgCol * imgProps.height) / imgProps.width;

  // Image box
  setColor(doc, [15, 23, 42], "fill");
  doc.rect(imgX, y, imgCol, imgH, "F");
  doc.addImage(imageDataUrl, "JPEG", imgX, y, imgCol, imgH);

  setColor(doc, GRAY, "text");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.text("Chest X-Ray (uploaded image)", imgX + imgCol / 2, y + imgH + 5, { align: "center" });

  // Results column
  const normalConf = result.confidences.find(c => c.label === "NORMAL")?.confidence ?? 0;
  const pneumoniaConf = result.confidences.find(c => c.label === "PNEUMONIA")?.confidence ?? 0;
  const isPneumonia = result.label === "PNEUMONIA";

  let ry = y;

  // Result badge
  const badgeColor = isPneumonia ? RED : GREEN;
  setColor(doc, badgeColor, "fill");
  doc.roundedRect(resX, ry, resCol, 18, 2, 2, "F");
  setColor(doc, WHITE, "text");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.text(result.label, resX + resCol / 2, ry + 8, { align: "center" });
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.text("AI PREDICTION", resX + resCol / 2, ry + 14, { align: "center" });
  ry += 24;

  // Confidence bars
  setColor(doc, DARK, "text");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.text("Confidence Scores", resX, ry);
  ry += 7;

  drawConfBar(doc, resX, ry, resCol, "NORMAL", normalConf, GREEN);
  ry += 14;
  drawConfBar(doc, resX, ry, resCol, "PNEUMONIA", pneumoniaConf, RED);
  ry += 20;

  // Interpretation note
  setColor(doc, GRAY, "text");
  doc.setFont("helvetica", "italic");
  doc.setFontSize(7.5);
  const note = isPneumonia
    ? "Findings are consistent with patterns seen in pneumonia cases. Confidence is high for the PNEUMONIA class."
    : "No significant opacities or consolidation detected. Confidence is high for the NORMAL class.";
  const noteLines = doc.splitTextToSize(note, resCol);
  doc.text(noteLines, resX, ry);

  y = Math.max(y + imgH + 10, ry + noteLines.length * 4 + 6);

  // ── Divider ─────────────────────────────────────────────────────────────────
  setColor(doc, [226, 232, 240], "draw");
  doc.setLineWidth(0.3);
  doc.line(14, y, W - 14, y);
  y += 8;

  // ── How the model works ─────────────────────────────────────────────────────
  setColor(doc, DARK, "text");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.text("About the Model", 14, y);
  y += 6;

  setColor(doc, GRAY, "text");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  const modelText =
    "This analysis was performed by a DenseNet121 convolutional neural network fine-tuned on the Kaggle " +
    "Chest X-Ray Images (Pneumonia) dataset (5,856 images, 2 classes). The model applies transfer learning " +
    "from ImageNet pre-trained weights and achieves strong performance on binary classification of NORMAL vs. " +
    "PNEUMONIA cases. It is deployed as a Gradio application on Hugging Face Spaces.";
  const modelLines = doc.splitTextToSize(modelText, W - 28);
  doc.text(modelLines, 14, y);
  y += modelLines.length * 4.5 + 6;

  // ── Disclaimer box ──────────────────────────────────────────────────────────
  setColor(doc, [255, 251, 235], "fill");
  setColor(doc, [217, 119, 6], "draw");
  doc.setLineWidth(0.4);
  const disclaimerText =
    "DISCLAIMER: This tool is intended for educational and demonstration purposes only. " +
    "It is NOT a certified medical device and must NOT be used as a substitute for professional " +
    "medical diagnosis, advice, or treatment. Always consult a qualified and licensed healthcare " +
    "provider regarding any medical condition. The authors accept no liability for decisions made " +
    "based on outputs from this tool.";
  const disclaimerLines = doc.splitTextToSize(disclaimerText, W - 40);
  const boxH = disclaimerLines.length * 4.5 + 10;
  doc.rect(14, y, W - 28, boxH, "FD");

  setColor(doc, [146, 64, 14], "text");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.text("IMPORTANT DISCLAIMER", 20, y + 6);
  doc.setFont("helvetica", "normal");
  doc.text(disclaimerLines, 20, y + 12);
  y += boxH + 8;

  // ── Footer ──────────────────────────────────────────────────────────────────
  const pageH = 297;
  setColor(doc, LIGHT_GRAY, "fill");
  doc.rect(0, pageH - 16, W, 16, "F");
  setColor(doc, GRAY, "text");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.text("Built by Hidayat Ullah  |  BS Artificial Intelligence, PAF-IAST", W / 2, pageH - 7, { align: "center" });
  doc.text("PneumoScan AI  •  Powered by DenseNet121 + Hugging Face Spaces", W / 2, pageH - 3, { align: "center" });

  // ── Save ────────────────────────────────────────────────────────────────────
  const safeName = fileName.replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]/gi, "_");
  doc.save(`pneumoscan_report_${safeName}_${Date.now()}.pdf`);
}

function drawConfBar(
  doc: jsPDF,
  x: number,
  y: number,
  width: number,
  label: string,
  value: number,
  color: readonly [number, number, number]
) {
  setColor(doc, DARK, "text");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  doc.text(label, x, y + 4);

  const pctStr = `${(value * 100).toFixed(1)}%`;
  doc.setFont("helvetica", "bold");
  doc.text(pctStr, x + width, y + 4, { align: "right" });

  const trackH = 4;
  const trackY = y + 7;
  setColor(doc, [226, 232, 240], "fill");
  doc.roundedRect(x, trackY, width, trackH, 1, 1, "F");
  if (value > 0) {
    setColor(doc, color, "fill");
    doc.roundedRect(x, trackY, width * value, trackH, 1, 1, "F");
  }
}

function loadImageProps(
  doc: jsPDF,
  dataUrl: string
): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => resolve({ width: 1, height: 1 });
    img.src = dataUrl;
    // jsPDF also needs it registered — call getImageProperties for side effect
    try { doc.getImageProperties(dataUrl); } catch { /* ignore */ }
  });
}
