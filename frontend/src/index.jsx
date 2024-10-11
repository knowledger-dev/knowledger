import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css"; // Import Tailwind CSS
import App from "./App";
import { AuthContextProvider } from "./context/AuthContext";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AuthContextProvider>
      <App />
    </AuthContextProvider>
  </React.StrictMode>
);
