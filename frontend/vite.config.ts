import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { TanStackRouterVite } from "@tanstack/router-vite-plugin";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

export default defineConfig({
  plugins: [
    TanStackRouterVite({ routesDirectory: "src/routes", generatedRouteTree: "src/routeTree.gen.ts" }),
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@worldscan/shared": path.resolve(__dirname, "../shared"),
    },
  },
  server: {
    port: 5173,
    // Proxy target: localhost:8000 for a natively-run API, or http://api:8000
    // when running in Docker (set VITE_PROXY_TARGET in docker-compose).
    proxy: (() => {
      const target = process.env.VITE_PROXY_TARGET || "http://localhost:8000";
      return {
        "/api": target,
        // /data/* serves per-project artifacts (mesh.ply, frames, thumbnails)
        // from the backend's StaticFiles mount. Without this, Vite returns the
        // SPA shell and the PLY loader silently fails.
        "/data": target,
      };
    })(),
  },
});
