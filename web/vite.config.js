import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite の設定。React プラグインを使うだけのシンプル構成。
export default defineConfig({
  plugins: [react()],
  server: {
    open: true, // npm run dev でブラウザを自動で開く
  },
});
