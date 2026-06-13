import React, { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { UploadCloud, FileImage, X, AlertCircle, Loader2, CheckCircle2, ShieldCheck, Download } from "lucide-react";
import { exportReportPDF } from "@/lib/export-pdf";
import { analyzeImage, type PredictionResult } from "@/lib/api";

export function UploadAnalyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = useCallback((selectedFile: File) => {
    if (selectedFile && (selectedFile.type === "image/jpeg" || selectedFile.type === "image/png")) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    } else {
      setError("Please select a valid JPG or PNG image.");
    }
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  }, [handleFileChange]);

  const clearSelection = useCallback(() => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setStatusMessage("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  // Make loadSample available globally so the SampleImages component can use it.
  // Accepts either a File object (new) or a URL string (legacy).
  useEffect(() => {
    (window as any).loadSampleToAnalyzer = async (fileOrUrl: File | string) => {
      try {
        clearSelection();
        let f: File;
        if (fileOrUrl instanceof File) {
          f = fileOrUrl;
        } else {
          const response = await fetch(fileOrUrl);
          const blob = await response.blob();
          const ext = blob.type === "image/png" ? "png" : "jpg";
          f = new File([blob], `sample-xray.${ext}`, { type: blob.type || "image/jpeg" });
        }
        handleFileChange(f);
        // Delay slightly to let state update visually before auto-clicking
        setTimeout(() => {
          const btn = document.getElementById("analyze-btn");
          if (btn) btn.click();
        }, 300);
      } catch (err) {
        console.error("Failed to load sample:", err);
      }
    };
    return () => {
      delete (window as any).loadSampleToAnalyzer;
    };
  }, [clearSelection, handleFileChange]);


  const fileToBase64 = (f: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(f);
    });
  };

  const startAnalysis = async () => {
    if (!file) return;
    
    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    // Safety timeout of 60 seconds
    const timeoutId = setTimeout(() => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    }, 60000);

    try {
      const res = await analyzeImage(file, setStatusMessage, signal);
      setResult(res);
      setStatusMessage("Analysis complete.");
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setError(err.message || "An error occurred during analysis.");
      } else {
        setError("The AI model is warming up. Please wait 30 seconds and try again.");
      }
    } finally {
      clearTimeout(timeoutId);
      setIsAnalyzing(false);
      abortControllerRef.current = null;
    }
  };

  const handleExportPDF = useCallback(async () => {
    if (!result || !previewUrl || !file) return;
    setIsExporting(true);
    try {
      await exportReportPDF(previewUrl, file.name, result);
    } catch (err) {
      console.error("PDF export failed:", err);
    } finally {
      setIsExporting(false);
    }
  }, [result, previewUrl, file]);

  const normalConf = result?.confidences.find(c => c.label === "NORMAL")?.confidence || 0;
  const pneumoniaConf = result?.confidences.find(c => c.label === "PNEUMONIA")?.confidence || 0;

  return (
    <section id="analyze" className="py-16 px-4 bg-slate-50">
      <div className="container mx-auto max-w-3xl">
        <Card className="shadow-xl border-slate-200 overflow-hidden">
          <CardHeader className="bg-white border-b border-slate-100 pb-6">
            <CardTitle className="text-2xl text-center">Diagnostic Analysis</CardTitle>
            <CardDescription className="text-center text-base">
              Upload a frontal chest X-ray image for AI evaluation
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6 md:p-8 bg-slate-50/50">
            
            {!file ? (
              <div 
                className={`border-2 border-dashed rounded-xl p-10 text-center transition-all duration-200 cursor-pointer flex flex-col items-center justify-center min-h-[300px]
                  ${isDragging ? 'border-primary bg-primary/5' : 'border-slate-300 hover:border-primary/50 hover:bg-slate-100'}`}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="bg-white p-4 rounded-full shadow-sm mb-4">
                  <UploadCloud className="w-8 h-8 text-slate-500" />
                </div>
                <h3 className="font-semibold text-lg text-slate-700 mb-1">Drag and drop X-ray image here</h3>
                <p className="text-sm text-slate-500 mb-4">or click to browse from your computer</p>
                <span className="text-xs text-slate-400 font-medium">Supports JPG, PNG (Max 5MB)</span>
              </div>
            ) : (
              <div className="flex flex-col md:flex-row gap-8 items-start">
                <div className="w-full md:w-1/2 flex flex-col items-center">
                  <div className="relative w-full rounded-xl overflow-hidden border border-slate-200 shadow-sm bg-black aspect-square flex items-center justify-center group">
                    <img src={previewUrl!} alt="X-ray preview" className="max-w-full max-h-full object-contain" />
                    {!isAnalyzing && !result && (
                       <button 
                         onClick={clearSelection}
                         className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 text-white p-1.5 rounded-full backdrop-blur-sm transition-colors"
                       >
                         <X className="w-4 h-4" />
                       </button>
                    )}
                  </div>
                  <div className="mt-3 text-sm font-medium text-slate-600 flex items-center gap-2">
                    <FileImage className="w-4 h-4" />
                    <span className="truncate max-w-[200px]">{file.name}</span>
                  </div>
                </div>

                <div className="w-full md:w-1/2 flex flex-col justify-center min-h-[300px]">
                  {!result && !isAnalyzing && !error && (
                    <div className="flex flex-col items-center justify-center text-center h-full text-slate-500">
                      <ShieldCheck className="w-12 h-12 text-slate-300 mb-4" />
                      <p className="mb-6">Image ready for processing. The model will analyze patterns consistent with pneumonia.</p>
                      <Button id="analyze-btn" size="lg" className="w-full shadow-md" onClick={startAnalysis}>
                        Analyze X-Ray
                      </Button>
                    </div>
                  )}

                  {isAnalyzing && (
                    <div className="flex flex-col items-center justify-center text-center h-full space-y-6">
                      <Loader2 className="w-10 h-10 animate-spin text-primary" />
                      <div className="space-y-2">
                        <h4 className="font-semibold text-lg">Analyzing Image...</h4>
                        <p className="text-sm text-slate-500 max-w-xs">{statusMessage}</p>
                      </div>
                      <Progress value={undefined} className="w-full max-w-[200px] h-2" />
                    </div>
                  )}

                  {error && (
                    <div className="flex flex-col items-center justify-center text-center h-full text-destructive p-4 bg-destructive/5 rounded-lg border border-destructive/20">
                      <AlertCircle className="w-10 h-10 mb-3" />
                      <p className="font-medium mb-4">{error}</p>
                      <Button variant="outline" onClick={clearSelection}>Try Again</Button>
                    </div>
                  )}

                  {result && (
                    <div className="flex flex-col h-full animate-in fade-in slide-in-from-right-4 duration-500">
                      <div className="mb-6 text-center md:text-left">
                        <p className="text-sm text-slate-500 font-medium uppercase tracking-wider mb-2">Diagnosis Prediction</p>
                        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border ${
                          result.label === "NORMAL" 
                            ? "bg-green-50 border-green-200 text-green-700" 
                            : "bg-red-50 border-red-200 text-red-700"
                        }`}>
                          {result.label === "NORMAL" ? <CheckCircle2 className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
                          <span className="text-xl font-bold tracking-tight">{result.label}</span>
                        </div>
                      </div>

                      <div className="space-y-5 flex-grow">
                        <div>
                          <div className="flex justify-between text-sm font-medium mb-1.5">
                            <span className="text-slate-700">Normal</span>
                            <span className="text-slate-500">{(normalConf * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2.5 w-full bg-slate-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-green-500 transition-all duration-1000 ease-out" 
                              style={{ width: `${normalConf * 100}%` }}
                            />
                          </div>
                        </div>

                        <div>
                          <div className="flex justify-between text-sm font-medium mb-1.5">
                            <span className="text-slate-700">Pneumonia</span>
                            <span className="text-slate-500">{(pneumoniaConf * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2.5 w-full bg-slate-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-red-500 transition-all duration-1000 ease-out" 
                              style={{ width: `${pneumoniaConf * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>

                      <div className="mt-8 pt-4 border-t border-slate-200 flex flex-col gap-2">
                        <Button
                          className="w-full gap-2"
                          onClick={handleExportPDF}
                          disabled={isExporting}
                        >
                          {isExporting ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Download className="w-4 h-4" />
                          )}
                          {isExporting ? "Generating PDF..." : "Export PDF Report"}
                        </Button>
                        <Button variant="outline" className="w-full" onClick={clearSelection}>
                          Analyze Another Image
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="image/jpeg,image/png" 
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  handleFileChange(e.target.files[0]);
                }
              }}
            />
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
