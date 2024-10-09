import { app, BrowserWindow, globalShortcut } from "electron";
import path from "path";
import process from "process";
let mainWindow;

app.whenReady().then(() => {
  const __dirname = path.resolve();
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true, // Ensure this matches your app setup
      contextIsolation: false, // Ensure this matches your app setup
      enableRemoteModule: true, // Ensure this matches your app setup
      preload: path.join(__dirname, "preload.js"), // Path to your preload script
    },
  });

  mainWindow.loadURL("http://localhost:5173"); // Assuming your React app runs here

  // Register a global shortcut listener
  globalShortcut.register("CommandOrControl+Shift+Space", () => {
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
    mainWindow = new BrowserWindow({
      width: 800,
      height: 600,
      webPreferences: {
        nodeIntegration: true, // Ensure this matches your app setup
        contextIsolation: false, // Ensure this matches your app setup
        enableRemoteModule: true, // Ensure this matches your app setup
        preload: path.join("electron", "preload.js"), // Path to your preload script
      },
    });

    mainWindow.loadURL("http://localhost:5173"); // Assuming your React app runs here
  }
});

// app.on("will-quit", () => {
//   // Unregister all shortcuts
//   globalShortcut.unregisterAll();
// });

// Old code

// const { app, BrowserWindow, globalShortcut } = require('electron');
// const path = require('path');

// function createWindow() {
//   const win = new BrowserWindow({
//     width: 800,
//     height: 600,
//     webPreferences: {
//       preload: path.join(__dirname, 'preload.js'),
//     },
//   });

//   if (process.env.NODE_ENV === 'development') {
//     win.loadURL('http://localhost:3000');
//   } else {
//     win.loadFile(path.join(__dirname, '../dist/index.html'));
//   }

//   return win;
// }

// app.whenReady().then(() => {
//   const win = createWindow();

//   // Global shortcut for Cmd + Space (macOS) or Ctrl + Space (Windows/Linux)
//   globalShortcut.register('CommandOrControl+Space', () => {
//     if (win.isMinimized()) win.restore();
//     win.focus();
//   });

//   app.on('activate', () => {
//     if (BrowserWindow.getAllWindows().length === 0) createWindow();
//   });
// });

// app.on('window-all-closed', () => {
//   if (process.platform !== 'darwin') {
//     app.quit();
//   }
// });

// app.on('will-quit', () => {
//   globalShortcut.unregisterAll();
// });
