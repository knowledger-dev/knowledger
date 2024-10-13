import { app, BrowserWindow, globalShortcut } from "electron";
import process from "process";
import * as CONSTANTS from "./ELECTRON_VARS.js";

let mainWindow;

const createWindow = () => {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true, // Ensure this matches your app setup
      contextIsolation: false, // Ensure this matches your app setup
      enableRemoteModule: true, // Ensure this matches your app setup
    },
    autoHideMenuBar: true,
  });

  // Load your index.html file which will contain the React app.
  // In production, you should load the built index.html file (from /build or /dist)
  // mainWindow.loadFile(path.join(__dirname, "dist/index.html"));
  mainWindow.loadURL(CONSTANTS.FRONTEND_HOST);

  // Open the DevTools. Turn off on deployment!!
  // mainWindow.webContents.openDevTools();
};

app.whenReady().then(() => {
  createWindow();

  // Register a global shortcut listener
  globalShortcut.register("Control+Shift+Space", () => {
    // Bring the window to the front
    if (mainWindow) {
      mainWindow.show();
      mainWindow.focus();
      // Send a message to the React app
      mainWindow.webContents.send("focus-input");
    }
  });
});

app.on("window-all-closed", () => {
  try {
    const platform = process.platform; // or the object you're trying to access
    if (platform !== "darwin") {
      // Unregister all shortcuts
      globalShortcut.unregisterAll();

      app.quit();
    }
  } catch (error) {
    console.error("Error accessing platform:", error);
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
