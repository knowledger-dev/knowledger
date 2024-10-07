import { app, BrowserWindow, globalShortcut } from "electron";

let mainWindow;

app.whenReady().then(() => {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true, // Ensure this matches your app setup
      contextIsolation: false, // Ensure this matches your app setup
    },
  });

  mainWindow.loadURL("http://localhost:5173"); // Assuming your React app runs here

  // Register a global shortcut listener
  globalShortcut.register("Control+Space", () => {
    // Bring the window to the front
    if (mainWindow) {
      mainWindow.show();
      mainWindow.focus();
      // Send a message to the React app
      mainWindow.webContents.send("focus-input");
    }
  });

  // // Listen for window resize event and reload the content
  // // TODO: This is not the best way to handle window resize, as the app will reload, change later
  // mainWindow.on("resize", () => {
  //   mainWindow.reload();
  // });

  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
      app.quit();
    }
  });
});

app.on("will-quit", () => {
  // Unregister all shortcuts
  globalShortcut.unregisterAll();
});

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
