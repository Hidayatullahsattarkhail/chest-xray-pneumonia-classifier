import { useState, useEffect } from "react";
import { X, AlertTriangle } from "lucide-react";

const STORAGE_KEY = "pneumoscan-disclaimer-dismissed";

export function DisclaimerBanner() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const dismissed = localStorage.getItem(STORAGE_KEY);
    if (!dismissed) setVisible(true);
  }, []);

  const dismiss = () => {
    localStorage.setItem(STORAGE_KEY, "1");
    setVisible(false);
  };

  if (!visible) return null;

  return (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-2.5">
      <div className="container mx-auto max-w-6xl flex items-center gap-3">
        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
        <p className="text-sm text-amber-800 flex-1">
          <span className="font-semibold">Educational demo only</span> — not for clinical use.
          Always consult a licensed physician before making any medical decisions.
        </p>
        <button
          onClick={dismiss}
          aria-label="Dismiss disclaimer"
          className="text-amber-600 hover:text-amber-800 transition-colors p-1 rounded hover:bg-amber-100"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
