import { contextBridge } from "electron";

// Safely expose process.platform to the renderer
contextBridge.exposeInMainWorld("electronAPI", {
  platform: process.platform,
});
