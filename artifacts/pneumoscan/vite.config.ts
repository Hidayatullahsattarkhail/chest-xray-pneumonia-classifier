import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig(async ({ command }) => {
  const rawPort = process.env.PORT;
  const port = rawPort && !Number.isNaN(Number(rawPort)) && Number(rawPort) > 0
    ? Number(rawPort)
    : 5173;

  const basePath = process.env.BASE_PATH ?? "/";

  const plugins = [
    react(),
    tailwindcss(),
  ];

  if (process.env.NODE_ENV !== "production" && process.env.REPL_ID !== undefined) {
    const { default: runtimeErrorOverlay } = await import("@replit/vite-plugin-runtime-error-modal");
    plugins.push(runtimeErrorOverlay());

    const { cartographer } = await import("@replit/vite-plugin-cartographer");
    plugins.push(cartographer({ root: path.resolve(import.meta.dirname, "..") }));

    const { devBanner } = await import("@replit/vite-plugin-dev-banner");
    plugins.push(devBanner());
  }

  return {
    base: basePath,
    plugins,
    resolve: {
      alias: {
        "@": path.resolve(import.meta.dirname, "src"),
        "@assets": path.resolve(import.meta.dirname, "..", "..", "attached_assets"),
      },
      dedupe: ["react", "react-dom"],
    },
    root: path.resolve(import.meta.dirname),
    build: {
      outDir: path.resolve(import.meta.dirname, "dist/public"),
      emptyOutDir: true,
    },
    ...(command === "serve"
      ? {
          server: {
            port,
            strictPort: true,
            host: "0.0.0.0",
            allowedHosts: true,
            fs: { strict: true },
          },
          preview: {
            port,
            host: "0.0.0.0",
            allowedHosts: true,
          },
        }
      : {}),
  };
});
