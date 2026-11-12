const { app, BrowserWindow, nativeTheme } = require("electron");
const path = require("path");

nativeTheme.themeSource = "dark";

function createWindow() {
  const win = new BrowserWindow({
    width: 1100,
    height: 820,
    minWidth: 420,
    minHeight: 600,
    title: "Finger Shapes",
    backgroundColor: "#202020",
    autoHideMenuBar: true,
  });

  win.loadFile(path.join(__dirname, "index.html"));
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  app.quit();
});